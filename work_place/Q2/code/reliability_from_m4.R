args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: reliability_from_m4.R model_data.csv output_dir")
}

data_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(20260824)

warnings_seen <- character()
write_error <- function(message) {
  writeLines(as.character(message), file.path(output_dir, "error.txt"), useBytes = TRUE)
}

tryCatch({
  suppressPackageStartupMessages(library(mgcv))
  data <- read.csv(
    data_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8-BOM"
  )
  required <- c(
    "row_id", "patient_id", "blood_draw_no", "GA", "Y", "BMI", "AGE",
    "gravidity_cat", "parity", "conception_mode"
  )
  missing <- setdiff(required, names(data))
  if (length(missing)) stop(paste("missing columns:", paste(missing, collapse = ", ")))
  if (nrow(data) != 1022 || length(unique(data$patient_id)) != 267) {
    stop("Q1 contract failed: expected 1022 observations and 267 patients")
  }

  means <- vapply(data[c("GA", "BMI", "AGE")], mean, numeric(1))
  data$GA_c <- data$GA - means[["GA"]]
  data$BMI_c <- data$BMI - means[["BMI"]]
  data$AGE_c <- data$AGE - means[["AGE"]]
  data$patient_id <- factor(data$patient_id)
  data$gravidity_cat <- factor(data$gravidity_cat, levels = c("1", "2", "3plus"))
  data$conception_mode <- factor(data$conception_mode, levels = c("natural", "assisted"))
  if (anyNA(data$gravidity_cat) || anyNA(data$conception_mode)) {
    stop("unknown Q1 clinical factor level")
  }

  model_formula <- Y ~ s(GA_c, k = 5) + s(BMI_c, k = 5) +
    ti(GA_c, BMI_c, k = c(5, 5)) + AGE_c +
    gravidity_cat + parity + conception_mode + s(patient_id, bs = "re")
  fit <- withCallingHandlers(
    gam(
      model_formula,
      data = data,
      family = betar(link = "logit"),
      method = "REML",
      select = TRUE
    ),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  conditional_fitted <- as.numeric(predict(fit, newdata = data, type = "response"))
  residual <- data$Y - conditional_fitted
  conditional_r2 <- 1 - sum(residual^2) / sum((data$Y - mean(data$Y))^2)
  conditional_rmse <- sqrt(mean(residual^2))
  conditional_mae <- mean(abs(residual))
  if (!isTRUE(fit$converged) || abs(conditional_r2 - 0.8263662212036017) > 1e-5) {
    stop(sprintf("Q1 sanity check failed: converged=%s conditional R2=%.12f", fit$converged, conditional_r2))
  }

  vc <- NULL
  invisible(capture.output(vc <- gam.vcomp(fit)))
  sigma_u2 <- as.numeric(vc["s(patient_id)", "std.dev"])^2
  precision <- fit$family$getTheta(TRUE)

  ordered <- data[order(data$patient_id, data$blood_draw_no, data$GA), ]
  first_index <- !duplicated(ordered$patient_id)
  profiles <- ordered[first_index, c(
    "patient_id", "BMI", "AGE", "gravidity_cat", "parity", "conception_mode"
  )]
  profiles$patient_id <- as.character(profiles$patient_id)
  if (nrow(profiles) != 267) stop("patient-profile extraction did not produce 267 rows")

  reference_patient <- levels(data$patient_id)[[1]]
  reference_newdata <- function(ga, bmi) {
    frame <- data.frame(
      GA = ga,
      BMI = bmi,
      GA_c = ga - means[["GA"]],
      BMI_c = bmi - means[["BMI"]],
      AGE_c = 0,
      gravidity_cat = factor("1", levels = levels(data$gravidity_cat)),
      parity = 0,
      conception_mode = factor("natural", levels = levels(data$conception_mode)),
      patient_id = factor(reference_patient, levels = levels(data$patient_id))
    )
    frame
  }

  ga_values <- seq(10, 25, by = 0.1)
  patient_surface <- merge(
    profiles[c("patient_id", "BMI")],
    data.frame(GA = ga_values),
    by = NULL,
    sort = FALSE
  )
  patient_reference <- reference_newdata(patient_surface$GA, patient_surface$BMI)
  patient_surface$eta_base <- as.numeric(predict(
    fit, newdata = patient_reference, type = "link", exclude = "s(patient_id)"
  ))
  write.csv(
    patient_surface[c("patient_id", "BMI", "GA", "eta_base")],
    file.path(output_dir, "patient_eta_surface.csv"), row.names = FALSE
  )

  plot_bmi <- seq(floor(min(profiles$BMI) * 10) / 10, ceiling(max(profiles$BMI) * 10) / 10, by = 0.1)
  plot_surface <- expand.grid(GA = ga_values, BMI = plot_bmi)
  plot_reference <- reference_newdata(plot_surface$GA, plot_surface$BMI)
  plot_surface$eta_base <- as.numeric(predict(
    fit, newdata = plot_reference, type = "link", exclude = "s(patient_id)"
  ))
  write.csv(plot_surface, file.path(output_dir, "plot_eta_surface.csv"), row.names = FALSE)

  offset_newdata <- reference_newdata(rep(means[["GA"]], nrow(profiles)), rep(means[["BMI"]], nrow(profiles)))
  offset_newdata$AGE_c <- profiles$AGE - means[["AGE"]]
  offset_newdata$gravidity_cat <- factor(profiles$gravidity_cat, levels = levels(data$gravidity_cat))
  offset_newdata$parity <- profiles$parity
  offset_newdata$conception_mode <- factor(profiles$conception_mode, levels = levels(data$conception_mode))
  reference_eta <- as.numeric(predict(
    fit,
    newdata = reference_newdata(means[["GA"]], means[["BMI"]]),
    type = "link",
    exclude = "s(patient_id)"
  ))
  profiles$z_offset <- as.numeric(predict(
    fit, newdata = offset_newdata, type = "link", exclude = "s(patient_id)"
  )) - reference_eta
  write.csv(profiles, file.path(output_dir, "patient_profiles.csv"), row.names = FALSE)

  write.csv(
    data.frame(
      row_id = as.character(data$row_id),
      patient_id = as.character(data$patient_id),
      observed_y = data$Y,
      conditional_fitted = conditional_fitted,
      residual = residual
    ),
    file.path(output_dir, "conditional_fit.csv"), row.names = FALSE
  )

  sm <- summary(fit)
  info <- data.frame(
    key = c(
      "k", "family", "method", "select", "converged", "outer_convergence",
      "n_obs", "n_patients", "conditional_r2", "conditional_rmse", "conditional_mae",
      "precision", "random_intercept_variance", "GA_mean", "BMI_mean", "AGE_mean",
      "deviance_explained", "warnings"
    ),
    value = c(
      5, "betar(logit)", "REML", TRUE, fit$converged,
      if (!is.null(fit$outer.info$conv)) fit$outer.info$conv else "",
      nrow(data), nlevels(data$patient_id), conditional_r2, conditional_rmse, conditional_mae,
      precision, sigma_u2, means[["GA"]], means[["BMI"]], means[["AGE"]],
      sm$dev.expl, paste(unique(warnings_seen), collapse = " | ")
    )
  )
  write.csv(info, file.path(output_dir, "model_info.csv"), row.names = FALSE)

  kc <- k.check(fit)
  kc_frame <- as.data.frame(kc)
  kc_frame$smooth_term <- rownames(kc_frame)
  rownames(kc_frame) <- NULL
  write.csv(kc_frame, file.path(output_dir, "k_check.csv"), row.names = FALSE)
  capture.output(gam.check(fit), file = file.path(output_dir, "gam_check.txt"))
}, error = function(e) {
  write_error(conditionMessage(e))
  quit(status = 1)
})
