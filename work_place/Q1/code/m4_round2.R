args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("usage: m4_round2.R k train.csv validation.csv output_dir make_grid")
}

k_value <- as.integer(args[[1]])
train_path <- args[[2]]
validation_path <- args[[3]]
output_dir <- args[[4]]
make_grid <- identical(args[[5]], "1")
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
  train$gravidity_cat <- factor(train$gravidity_cat, levels = c("1", "2", "3plus"))
  train$conception_mode <- factor(train$conception_mode, levels = c("natural", "assisted"))
  validation$gravidity_cat <- factor(validation$gravidity_cat, levels = levels(train$gravidity_cat))
  validation$conception_mode <- factor(validation$conception_mode, levels = levels(train$conception_mode))
  if (anyNA(train$gravidity_cat) || anyNA(train$conception_mode) ||
      anyNA(validation$gravidity_cat) || anyNA(validation$conception_mode)) {
    stop("unknown clinical factor level")
  }

  model_formula <- Y ~ s(GA_c, k = k_value) + s(BMI_c, k = k_value) +
    ti(GA_c, BMI_c, k = c(k_value, k_value)) + AGE_c +
    gravidity_cat + parity + conception_mode + s(patient_id, bs = "re")

  fit <- withCallingHandlers(
    gam(
      model_formula,
      data = train,
      family = betar(link = "logit"),
      method = "REML",
      select = TRUE
    ),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  validation$patient_id <- factor(
    rep(levels(train$patient_id)[[1]], nrow(validation)),
    levels = levels(train$patient_id)
  )
  eta_fixed <- as.numeric(predict(
    fit, newdata = validation, type = "link", exclude = "s(patient_id)"
  ))
  write.csv(
    data.frame(
      row_id = validation$row_id,
      eta_fixed = eta_fixed,
      conditional_at_u0_prediction = plogis(eta_fixed)
    ),
    file.path(output_dir, "predictions.csv"), row.names = FALSE
  )

  train_eta_fixed <- as.numeric(predict(
    fit, newdata = train, type = "link", exclude = "s(patient_id)"
  ))
  train_eta_conditional <- as.numeric(predict(fit, newdata = train, type = "link"))
  write.csv(
    data.frame(
      row_id = train$row_id,
      eta_fixed = train_eta_fixed,
      eta_conditional = train_eta_conditional,
      mu_fixed = plogis(train_eta_fixed),
      mu_conditional = plogis(train_eta_conditional)
    ),
    file.path(output_dir, "training_components.csv"), row.names = FALSE
  )

  sm <- summary(fit)
  p_table <- as.data.frame(sm$p.table)
  if (nrow(p_table) > 0) {
    p_table$term <- rownames(p_table)
    rownames(p_table) <- NULL
    names(p_table)[seq_len(min(4, ncol(p_table) - 1))] <-
      c("estimate", "std_error", "statistic", "p_value")[seq_len(min(4, ncol(p_table) - 1))]
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
  sigma_u2 <- NA_real_
  if (!is.null(vc)) {
    vc_frame <- as.data.frame(vc)
    vc_frame$component <- rownames(vc_frame)
    rownames(vc_frame) <- NULL
    if ("std.dev" %in% names(vc_frame)) vc_frame$variance <- vc_frame$std.dev^2
    write.csv(vc_frame, file.path(output_dir, "variance_components.csv"), row.names = FALSE)
    if ("s(patient_id)" %in% rownames(vc) && "std.dev" %in% colnames(vc)) {
      sigma_u2 <- as.numeric(vc["s(patient_id)", "std.dev"])^2
    }
  }

  re_position <- which(vapply(fit$smooth, function(x) x$label == "s(patient_id)", logical(1)))
  if (length(re_position) == 1) {
    re_smooth <- fit$smooth[[re_position]]
    re_indices <- re_smooth$first.para:re_smooth$last.para
    write.csv(
      data.frame(patient_id = levels(train$patient_id), random_effect_estimate = coef(fit)[re_indices]),
      file.path(output_dir, "random_effect_estimates.csv"), row.names = FALSE
    )
  }

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

  precision <- fit$family$getTheta(TRUE)
  outer_convergence <- if (!is.null(fit$outer.info$conv)) fit$outer.info$conv else ""
  info <- data.frame(
    key = c(
      "k", "converged", "outer_convergence", "n_obs", "n_patients", "aic",
      "log_likelihood", "reml_score", "deviance_explained", "adjusted_r2",
      "precision", "random_intercept_variance", "total_edf", "warnings"
    ),
    value = c(
      k_value, fit$converged, outer_convergence, nrow(train), nlevels(train$patient_id),
      AIC(fit), as.numeric(logLik(fit)), fit$gcv.ubre, sm$dev.expl, sm$r.sq,
      precision, sigma_u2, sum(fit$edf), paste(unique(warnings_seen), collapse = " | ")
    )
  )
  write.csv(info, file.path(output_dir, "info.csv"), row.names = FALSE)

  if (make_grid) {
    ga_values <- seq(min(train$GA), max(train$GA), length.out = 60)
    bmi_values <- seq(min(train$BMI), max(train$BMI), length.out = 60)
    grid <- expand.grid(GA = ga_values, BMI = bmi_values)
    grid$GA_c <- grid$GA - train$GA_mean[[1]]
    grid$BMI_c <- grid$BMI - train$BMI_mean[[1]]
    grid$AGE_c <- 0
    grid$gravidity_cat <- factor("1", levels = levels(train$gravidity_cat))
    grid$parity <- 0
    grid$conception_mode <- factor("natural", levels = levels(train$conception_mode))
    grid$patient_id <- factor(
      rep(levels(train$patient_id)[[1]], nrow(grid)), levels = levels(train$patient_id)
    )
    grid$eta_fixed <- as.numeric(predict(
      fit, newdata = grid, type = "link", exclude = "s(patient_id)"
    ))
    write.csv(grid[, c("GA", "BMI", "eta_fixed")], file.path(output_dir, "effect_grid.csv"), row.names = FALSE)
  }
}, error = function(e) {
  write_error(conditionMessage(e))
  quit(status = 1)
})
