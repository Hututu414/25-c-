args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
  stop("usage: Rscript calibrate_z.R train.csv apply.csv train_out.csv apply_out.csv diagnostics.csv fold_tag")
}

suppressPackageStartupMessages(library(mgcv))

train <- read.csv(args[[1]], check.names = FALSE)
apply_data <- read.csv(args[[2]], check.names = FALSE)
normal <- train$is_normal_reference == 1
if (sum(normal) < 50L) stop("too few training-normal records for conditional calibration")

diagnostics <- list()
for (chromosome in c(13L, 18L, 21L)) {
  response <- paste0("Z", chromosome)
  gc_name <- paste0("GC", chromosome)
  formula_text <- paste0(
    response,
    " ~ s(GA,k=4) + s(BMI,k=4) + s(X_Z,k=4) + s(", gc_name,
    ",k=4) + s(GC_global,k=4) + alignment_rate + duplication_rate + log_unique_reads + filtered_rate"
  )
  model_formula <- as.formula(formula_text)
  family_used <- "scat"
  fit <- tryCatch(
    gam(model_formula, data = train[normal, ], family = scat(link = "identity"), method = "REML"),
    error = function(error) NULL
  )
  if (is.null(fit) || !isTRUE(fit$converged)) {
    family_used <- "gaussian_fallback"
    fit <- gam(model_formula, data = train[normal, ], family = gaussian(), method = "REML")
  }
  if (!isTRUE(fit$converged)) stop(paste("calibration GAM did not converge for", response))

  normal_prediction <- as.numeric(predict(fit, newdata = train[normal, ], type = "response"))
  residual <- train[normal, response] - normal_prediction
  robust_scale <- 1.4826 * median(abs(residual - median(residual)))
  scale_method <- "1.4826*MAD"
  if (!is.finite(robust_scale) || robust_scale <= 1e-8) {
    robust_scale <- sd(residual)
    scale_method <- "SD_fallback"
  }
  if (!is.finite(robust_scale) || robust_scale <= 1e-8) stop("invalid calibration scale")

  train_prediction <- as.numeric(predict(fit, newdata = train, type = "response"))
  apply_prediction <- as.numeric(predict(fit, newdata = apply_data, type = "response"))
  calibrated_name <- paste0(response, "_cal")
  train[[calibrated_name]] <- (train[[response]] - train_prediction) / robust_scale
  apply_data[[calibrated_name]] <- (apply_data[[response]] - apply_prediction) / robust_scale

  smooth_edf <- if (is.null(summary(fit)$s.table)) 0 else sum(summary(fit)$s.table[, "edf"])
  diagnostics[[length(diagnostics) + 1L]] <- data.frame(
    fold_tag = args[[6]], chromosome = paste0("T", chromosome), family = family_used,
    normal_n = sum(normal), robust_scale = robust_scale, scale_method = scale_method,
    residual_median = median(residual), residual_mad = median(abs(residual - median(residual))),
    smooth_edf = smooth_edf, converged = isTRUE(fit$converged), stringsAsFactors = FALSE
  )
}

write.csv(train, args[[3]], row.names = FALSE, fileEncoding = "UTF-8")
write.csv(apply_data, args[[4]], row.names = FALSE, fileEncoding = "UTF-8")
write.csv(do.call(rbind, diagnostics), args[[5]], row.names = FALSE, fileEncoding = "UTF-8")
