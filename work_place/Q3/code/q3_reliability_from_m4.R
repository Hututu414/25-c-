args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: q3_reliability_from_m4.R q3_model_data.csv output_dir")
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
    "row_id", "patient_id", "blood_draw_no", "GA", "Y", "BMI", "Height", "AGE",
    "gravidity_cat", "parity", "conception_mode"
  )
  missing <- setdiff(required, names(data))
  if (length(missing)) stop(paste("missing columns:", paste(missing, collapse = ", ")))
  if (nrow(data) != 1022 || length(unique(data$patient_id)) != 267 || anyNA(data[required])) {
    stop("Q3 data contract failed: expected 1022 complete observations and 267 patients")
  }

  means <- vapply(data[c("GA", "BMI", "Height", "AGE")], mean, numeric(1))
  data$GA_c <- data$GA - means[["GA"]]
  data$BMI_c <- data$BMI - means[["BMI"]]
  data$Height_c <- data$Height - means[["Height"]]
  data$AGE_c <- data$AGE - means[["AGE"]]
  data$patient_id <- factor(data$patient_id)
  data$gravidity_cat <- factor(data$gravidity_cat, levels = c("1", "2", "3plus"))
  data$conception_mode <- factor(data$conception_mode, levels = c("natural", "assisted"))
  if (anyNA(data$gravidity_cat) || anyNA(data$conception_mode)) {
    stop("unknown clinical factor level")
  }

  model_formula <- Y ~ s(GA_c, k = 5) + s(BMI_c, k = 5) +
    ti(GA_c, BMI_c, k = c(5, 5)) + s(Height_c, k = 4) + AGE_c +
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
  if (!isTRUE(fit$converged)) stop("Q3 Beta-GAMM did not converge")

  conditional_fitted <- as.numeric(predict(fit, newdata = data, type = "response"))
  residual <- data$Y - conditional_fitted
  conditional_r2 <- 1 - sum(residual^2) / sum((data$Y - mean(data$Y))^2)
  conditional_rmse <- sqrt(mean(residual^2))
  conditional_mae <- mean(abs(residual))

  vc <- NULL
  invisible(capture.output(vc <- gam.vcomp(fit)))
  sigma_u2 <- as.numeric(vc["s(patient_id)", "std.dev"])^2
  precision <- fit$family$getTheta(TRUE)
  sm <- summary(fit)

  ordered <- data[order(data$patient_id, data$blood_draw_no, data$GA), ]
  profiles <- ordered[!duplicated(ordered$patient_id), c(
    "patient_id", "BMI", "Height", "AGE", "gravidity_cat", "parity", "conception_mode"
  )]
  profiles$patient_id <- as.character(profiles$patient_id)
  if (nrow(profiles) != 267) stop("patient-profile extraction did not produce 267 rows")

  reference_patient <- levels(data$patient_id)[[1]]
  ga_values <- seq(10, 25, by = 0.1)
  surface <- merge(profiles, data.frame(GA = ga_values), by = NULL, sort = FALSE)
  surface$GA_c <- surface$GA - means[["GA"]]
  surface$BMI_c <- surface$BMI - means[["BMI"]]
  surface$Height_c <- surface$Height - means[["Height"]]
  surface$AGE_c <- surface$AGE - means[["AGE"]]
  surface$profile_patient_id <- surface$patient_id
  surface$patient_id <- factor(reference_patient, levels = levels(data$patient_id))
  surface$gravidity_cat <- factor(surface$gravidity_cat, levels = levels(data$gravidity_cat))
  surface$conception_mode <- factor(surface$conception_mode, levels = levels(data$conception_mode))
  surface$eta_base <- as.numeric(predict(
    fit, newdata = surface, type = "link", exclude = "s(patient_id)"
  ))
  write.csv(
    transform(
      surface[c("profile_patient_id", "BMI", "Height", "AGE", "gravidity_cat", "parity", "conception_mode", "GA", "eta_base")],
      patient_id = profile_patient_id
    )[c("patient_id", "BMI", "Height", "AGE", "gravidity_cat", "parity", "conception_mode", "GA", "eta_base")],
    file.path(output_dir, "patient_eta_surface.csv"), row.names = FALSE
  )
  write.csv(profiles, file.path(output_dir, "patient_profiles.csv"), row.names = FALSE)

  height_grid <- seq(min(data$Height), max(data$Height), length.out = 101)
  height_newdata <- data.frame(
    GA = means[["GA"]], BMI = means[["BMI"]], Height = height_grid,
    GA_c = 0, BMI_c = 0, Height_c = height_grid - means[["Height"]], AGE_c = 0,
    gravidity_cat = factor("1", levels = levels(data$gravidity_cat)), parity = 0,
    conception_mode = factor("natural", levels = levels(data$conception_mode)),
    patient_id = factor(reference_patient, levels = levels(data$patient_id))
  )
  height_term <- as.numeric(predict(
    fit, newdata = height_newdata, type = "terms", terms = "s(Height_c)"
  ))
  write.csv(
    data.frame(Height = height_grid, Height_c = height_grid - means[["Height"]], link_partial_effect = height_term),
    file.path(output_dir, "height_effect.csv"), row.names = FALSE
  )

  smooth_terms <- as.data.frame(sm$s.table)
  smooth_terms$smooth_term <- rownames(smooth_terms)
  rownames(smooth_terms) <- NULL
  write.csv(smooth_terms, file.path(output_dir, "smooth_terms.csv"), row.names = FALSE)

  parametric_terms <- as.data.frame(sm$p.table)
  parametric_terms$term <- rownames(parametric_terms)
  rownames(parametric_terms) <- NULL
  write.csv(parametric_terms, file.path(output_dir, "parametric_terms.csv"), row.names = FALSE)

  kc <- as.data.frame(k.check(fit))
  kc$smooth_term <- rownames(kc)
  rownames(kc) <- NULL
  write.csv(kc, file.path(output_dir, "k_check.csv"), row.names = FALSE)
  capture.output(gam.check(fit), file = file.path(output_dir, "gam_check.txt"))

  info <- data.frame(
    key = c(
      "formula", "family", "method", "select", "converged", "outer_convergence",
      "n_obs", "n_patients", "conditional_r2", "conditional_rmse", "conditional_mae",
      "precision", "random_intercept_variance", "GA_mean", "BMI_mean", "Height_mean", "AGE_mean",
      "deviance_explained", "warnings"
    ),
    value = c(
      paste(deparse(model_formula), collapse = " "), "betar(logit)", "REML", TRUE, fit$converged,
      if (!is.null(fit$outer.info$conv)) fit$outer.info$conv else "",
      nrow(data), nlevels(data$patient_id), conditional_r2, conditional_rmse, conditional_mae,
      precision, sigma_u2, means[["GA"]], means[["BMI"]], means[["Height"]], means[["AGE"]],
      sm$dev.expl, paste(unique(warnings_seen), collapse = " | ")
    )
  )
  write.csv(info, file.path(output_dir, "model_info.csv"), row.names = FALSE)
}, error = function(e) {
  write_error(conditionMessage(e))
  quit(status = 1)
})
