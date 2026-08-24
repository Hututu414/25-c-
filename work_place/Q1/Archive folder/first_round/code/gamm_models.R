args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("usage: gamm_models.R M3|M4 train.csv validation.csv output_dir")
}

model_id <- args[[1]]
train_path <- args[[2]]
validation_path <- args[[3]]
output_dir <- args[[4]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(20260824)

warnings_seen <- character()
write_error <- function(message) {
  writeLines(as.character(message), file.path(output_dir, "error.txt"), useBytes = TRUE)
}

tryCatch({
  suppressPackageStartupMessages(library(mgcv))
  train <- read.csv(train_path, stringsAsFactors = FALSE, check.names = FALSE)
  validation <- read.csv(validation_path, stringsAsFactors = FALSE, check.names = FALSE)
  train$patient_id <- factor(train$patient_id)
  train$logit_y <- qlogis(train$Y)

  if (model_id == "M3") {
    formula <- logit_y ~ s(GA_c, k = 5) + s(BMI_c, k = 5) + AGE_c +
      s(patient_id, bs = "re")
    family <- gaussian()
  } else if (model_id == "M4") {
    formula <- Y ~ s(GA_c, k = 5) + s(BMI_c, k = 5) +
      ti(GA_c, BMI_c, k = c(5, 5)) + AGE_c + s(patient_id, bs = "re")
    family <- betar(link = "logit")
  } else {
    stop("unknown model: ", model_id)
  }

  fit <- withCallingHandlers(
    gam(formula, data = train, family = family, method = "REML"),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  # New-patient prediction: the factor value is a harmless placeholder because
  # the patient random-effect smooth is explicitly excluded.
  validation$patient_id <- factor(
    rep(levels(train$patient_id)[[1]], nrow(validation)),
    levels = levels(train$patient_id)
  )
  pred <- predict(
    fit,
    newdata = validation,
    type = "response",
    exclude = "s(patient_id)"
  )
  if (model_id == "M3") pred <- plogis(pred)
  write.csv(
    data.frame(row_id = validation$row_id, predicted = as.numeric(pred)),
    file.path(output_dir, "predictions.csv"),
    row.names = FALSE
  )

  predict_fixed <- function(newdata) {
    newdata$patient_id <- factor(
      rep(levels(train$patient_id)[[1]], nrow(newdata)),
      levels = levels(train$patient_id)
    )
    value <- predict(fit, newdata = newdata, type = "response", exclude = "s(patient_id)")
    if (model_id == "M3") value <- plogis(value)
    as.numeric(value)
  }
  ga_values <- seq(min(train$GA), max(train$GA), length.out = 120)
  bmi_values <- as.numeric(quantile(train$BMI, c(0.25, 0.5, 0.75)))
  ga_grid <- do.call(rbind, lapply(bmi_values, function(bmi_value) {
    frame <- data.frame(
      GA = ga_values,
      BMI = bmi_value,
      GA_c = ga_values - train$GA_mean[[1]],
      BMI_c = bmi_value - train$BMI_mean[[1]],
      AGE_c = 0
    )
    frame$panel <- "GA"
    frame$curve <- sprintf("BMI = %.2f", bmi_value)
    frame$x_value <- frame$GA
    frame$predicted <- predict_fixed(frame)
    frame
  }))
  bmi_sequence <- seq(min(train$BMI), max(train$BMI), length.out = 120)
  ga_quantiles <- as.numeric(quantile(train$GA, c(0.25, 0.5, 0.75)))
  bmi_grid <- do.call(rbind, lapply(ga_quantiles, function(ga_value) {
    frame <- data.frame(
      GA = ga_value,
      BMI = bmi_sequence,
      GA_c = ga_value - train$GA_mean[[1]],
      BMI_c = bmi_sequence - train$BMI_mean[[1]],
      AGE_c = 0
    )
    frame$panel <- "BMI"
    frame$curve <- sprintf("GA = %.2f", ga_value)
    frame$x_value <- frame$BMI
    frame$predicted <- predict_fixed(frame)
    frame
  }))
  write.csv(
    rbind(ga_grid, bmi_grid)[, c("panel", "curve", "x_value", "predicted")],
    file.path(output_dir, "effect_predictions.csv"),
    row.names = FALSE
  )

  sm <- summary(fit)
  p_table <- as.data.frame(sm$p.table)
  if (nrow(p_table) > 0) {
    p_table$term <- rownames(p_table)
    rownames(p_table) <- NULL
    names(p_table)[seq_len(min(4, ncol(p_table) - 1))] <-
      c("estimate", "std_error", "statistic", "p_value")[seq_len(min(4, ncol(p_table) - 1))]
    if (all(c("estimate", "std_error") %in% names(p_table))) {
      p_table$CI_low <- p_table$estimate - qnorm(0.975) * p_table$std_error
      p_table$CI_high <- p_table$estimate + qnorm(0.975) * p_table$std_error
    }
    write.csv(p_table, file.path(output_dir, "coefficients.csv"), row.names = FALSE)
  }

  s_table <- as.data.frame(sm$s.table)
  if (nrow(s_table) > 0) {
    s_table$smooth_term <- rownames(s_table)
    rownames(s_table) <- NULL
    names(s_table)[seq_len(min(4, ncol(s_table) - 1))] <-
      c("edf", "reference_df", "statistic", "p_value")[seq_len(min(4, ncol(s_table) - 1))]
    write.csv(s_table, file.path(output_dir, "smooth_terms.csv"), row.names = FALSE)
  }

  vc <- NULL
  invisible(capture.output(vc <- gam.vcomp(fit)))
  if (!is.null(vc)) {
    vc_frame <- as.data.frame(vc)
    vc_frame$component <- rownames(vc_frame)
    rownames(vc_frame) <- NULL
    if ("std.dev" %in% names(vc_frame)) vc_frame$variance <- vc_frame$std.dev^2
    write.csv(vc_frame, file.path(output_dir, "random_effects.csv"), row.names = FALSE)
  }

  sp <- data.frame(smooth_term = names(fit$sp), smoothing_parameter = as.numeric(fit$sp))
  write.csv(sp, file.path(output_dir, "smoothing_parameters.csv"), row.names = FALSE)

  kc <- tryCatch(k.check(fit), error = function(e) NULL)
  if (!is.null(kc)) {
    kc_frame <- as.data.frame(kc)
    kc_frame$smooth_term <- rownames(kc_frame)
    rownames(kc_frame) <- NULL
    names(kc_frame)[seq_len(min(4, ncol(kc_frame) - 1))] <-
      c("k_prime", "edf", "k_index", "p_value")[seq_len(min(4, ncol(kc_frame) - 1))]
    write.csv(kc_frame, file.path(output_dir, "k_check.csv"), row.names = FALSE)
  }
  capture.output(gam.check(fit), file = file.path(output_dir, "gam_check.txt"))

  precision <- if (model_id == "M4") fit$family$getTheta(TRUE) else NA_real_
  info <- data.frame(
    key = c(
      "model", "converged", "n_obs", "n_patients", "aic", "bic", "log_likelihood",
      "reml_score", "deviance_explained", "adjusted_r2", "scale", "precision",
      "total_edf", "warnings"
    ),
    value = c(
      model_id, fit$converged, nrow(train), nlevels(train$patient_id), AIC(fit), BIC(fit),
      as.numeric(logLik(fit)), fit$gcv.ubre, sm$dev.expl, sm$r.sq, sm$scale, precision,
      sum(fit$edf), paste(unique(warnings_seen), collapse = " | ")
    )
  )
  write.csv(info, file.path(output_dir, "info.csv"), row.names = FALSE)
}, error = function(e) {
  write_error(conditionMessage(e))
  quit(status = 1)
})
