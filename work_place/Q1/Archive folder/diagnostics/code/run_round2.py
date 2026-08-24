from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_q1_models as round1
import statsmodels.api as sm
from numpy.polynomial.hermite import hermgauss
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize_scalar
from scipy.special import betaln, expit, logit, polygamma
from scipy.stats import beta as beta_distribution
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

SEED = 20260824
Q1_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Q1_ROOT.parent.parent
SOURCE_XLSX = PROJECT_ROOT / "C题" / "附件.xlsx"
DATA_DIR = Q1_ROOT / "data_processed"
ROUND2_DIR = Q1_ROOT / "outputs_round2"
RAW_DIR = ROUND2_DIR / "raw"
DECISION_DIR = ROUND2_DIR / "decision"
FIGURE_DIR = ROUND2_DIR / "figures"
PAYLOAD_PATH = ROUND2_DIR / ".artifact_payload.json"
M4_K_VALUES = (4, 5, 6)
QUADRATURE_NODES = 30
THRESHOLD = 0.04
BOUNDARY_LOW = 0.035
BOUNDARY_HIGH = 0.045


def self_check() -> None:
    eta = np.array([-3.0, -1.0, 0.0])
    marginal, _ = integrate_m4(eta, sigma_u2=0.0, phi=100.0)
    assert np.allclose(marginal, expit(eta))
    y = expit(eta + 0.5)
    u_hat, success, _, _ = estimate_random_intercept(y, eta, phi=200.0, sigma_u2=0.5)
    assert success and np.isfinite(u_hat) and u_hat > 0
    assert round1.parse_ga("16W+1") == 16 + 1 / 7


def load_round2_data() -> tuple[pd.DataFrame, dict[str, object]]:
    base = pd.read_csv(DATA_DIR / "q1_male_aggregated.csv")
    folds = pd.read_csv(DATA_DIR / "fold_assignments.csv")
    manifest = json.loads((DATA_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    if len(base) != manifest["aggregated_rows"] or base["patient_id"].nunique() != manifest["source_patients"]:
        raise AssertionError("first-round processed data no longer matches its manifest")
    check_folds = base[["patient_id", "fold"]].drop_duplicates().sort_values("patient_id").reset_index(drop=True)
    expected_folds = folds.sort_values("patient_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(check_folds, expected_folds, check_dtype=False)

    raw = pd.read_excel(SOURCE_XLSX, sheet_name="男胎检测数据", engine="openpyxl")
    raw.columns = [str(column).strip() for column in raw.columns]
    required = [
        "孕妇代码", "检测抽血次数", "检测孕周", "怀孕次数", "生产次数", "IVF妊娠"
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"source workbook misses round-two clinical columns: {missing}")

    clinical = raw[required].copy()
    clinical["GA"] = clinical["检测孕周"].map(round1.parse_ga)
    if clinical["GA"].isna().any():
        raise ValueError("round-two clinical merge found an invalid gestational age")
    patient_values = sorted(clinical["孕妇代码"].astype(str).unique())
    patient_map = {value: f"P{index:04d}" for index, value in enumerate(patient_values, 1)}
    clinical["patient_id"] = clinical["孕妇代码"].astype(str).map(patient_map)
    clinical["blood_draw_no"] = pd.to_numeric(clinical["检测抽血次数"], errors="raise")

    keys = ["patient_id", "blood_draw_no", "GA"]
    for column in ["怀孕次数", "生产次数", "IVF妊娠"]:
        if clinical.groupby(keys, sort=False)[column].nunique(dropna=False).gt(1).any():
            raise AssertionError(f"clinical field varies within a technical-repeat group: {column}")
    clinical = clinical.groupby(keys, as_index=False, sort=True).agg(
        gravidity_raw=("怀孕次数", "first"),
        parity=("生产次数", "first"),
        conception_raw=("IVF妊娠", "first"),
    )
    clinical["gravidity_cat"] = clinical["gravidity_raw"].astype(str).str.strip().replace({"≥3": "3plus"})
    if not set(clinical["gravidity_cat"]).issubset({"1", "2", "3plus"}):
        raise ValueError("unexpected gravidity category")
    clinical["parity"] = pd.to_numeric(clinical["parity"], errors="raise").astype(float)
    allowed_modes = {"自然受孕", "IUI（人工授精）", "IVF（试管婴儿）"}
    if not set(clinical["conception_raw"]).issubset(allowed_modes):
        raise ValueError("unexpected conception mode")
    clinical["conception_mode"] = np.where(
        clinical["conception_raw"].eq("自然受孕"), "natural", "assisted"
    )

    data = base.merge(
        clinical[keys + ["gravidity_cat", "parity", "conception_mode"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if data[["gravidity_cat", "parity", "conception_mode"]].isna().any().any():
        raise AssertionError("clinical enrichment failed for at least one biological observation")
    if data.groupby("patient_id")["fold"].nunique().max() != 1:
        raise AssertionError("patient-level fold leakage after clinical enrichment")
    data.to_csv(DATA_DIR / "q1_round2_model_data.csv", index=False, encoding="utf-8-sig")

    round2_manifest = {
        **manifest,
        "round2_rows": len(data),
        "round2_patients": data["patient_id"].nunique(),
        "fold_assignment": "reused byte-for-value from first-round fold_assignments.csv",
        "clinical_covariates": {
            "gravidity_cat": ["1", "2", "3plus"],
            "parity": "numeric count",
            "conception_mode": ["natural", "assisted (IUI/IVF combined)"],
        },
        "prohibited_post_test_variables_used": False,
    }
    (DATA_DIR / "round2_manifest.json").write_text(
        json.dumps(round2_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data, round2_manifest


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "RMSE": float(mean_squared_error(observed, predicted) ** 0.5),
        "MAE": float(mean_absolute_error(observed, predicted)),
        "R2": float(r2_score(observed, predicted)),
    }


def fold_metrics(predictions: pd.DataFrame, model: str, prediction_column: str) -> pd.DataFrame:
    rows = []
    for fold, frame in predictions.groupby("fold", sort=True):
        row = {"Model": model, "fold": int(fold), "N": len(frame)}
        row.update(regression_metrics(frame["observed_y"], frame[prediction_column]))
        boundary = frame[frame["observed_y"].between(BOUNDARY_LOW, BOUNDARY_HIGH)]
        row["Boundary_RMSE"] = (
            regression_metrics(boundary["observed_y"], boundary[prediction_column])["RMSE"]
            if len(boundary) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    truth = np.asarray(observed >= THRESHOLD, dtype=int)
    score = np.asarray(predicted, dtype=float)
    decision = np.asarray(score >= THRESHOLD, dtype=int)
    return {
        "Accuracy": float(accuracy_score(truth, decision)),
        "Precision": float(precision_score(truth, decision, zero_division=0)),
        "Recall": float(recall_score(truth, decision, zero_division=0)),
        "F1": float(f1_score(truth, decision, zero_division=0)),
        "AUC": float(roc_auc_score(truth, score)),
    }


def integrate_m4(
    eta_fixed: np.ndarray, sigma_u2: float, phi: float
) -> tuple[np.ndarray, np.ndarray]:
    eta = np.asarray(eta_fixed, dtype=float)
    sigma = math.sqrt(max(float(sigma_u2), 0.0))
    nodes, weights = hermgauss(QUADRATURE_NODES)
    random_effects = math.sqrt(2.0) * sigma * nodes
    mu = expit(eta[:, None] + random_effects[None, :])
    normalized_weights = weights / math.sqrt(math.pi)
    marginal_mean = mu @ normalized_weights
    alpha = np.clip(mu * phi, 1e-10, None)
    beta = np.clip((1.0 - mu) * phi, 1e-10, None)
    pass_probability = beta_distribution.sf(THRESHOLD, alpha, beta) @ normalized_weights
    return marginal_mean, pass_probability


def estimate_random_intercept(
    observed: np.ndarray, eta_fixed: np.ndarray, phi: float, sigma_u2: float
) -> tuple[float, bool, int, bool]:
    y = np.asarray(observed, dtype=float)
    eta = np.asarray(eta_fixed, dtype=float)
    sigma = math.sqrt(max(float(sigma_u2), 1e-12))
    bound = max(8.0 * sigma, 1.0)

    def objective(u: float) -> float:
        mu = np.clip(expit(eta + u), 1e-10, 1 - 1e-10)
        alpha = mu * phi
        beta = (1 - mu) * phi
        log_likelihood = np.sum(
            (alpha - 1) * np.log(y)
            + (beta - 1) * np.log1p(-y)
            - betaln(alpha, beta)
        )
        return float(-log_likelihood + u * u / (2 * max(sigma_u2, 1e-12)))

    # ponytail: bounded 1-D MAP is enough here; widen only if estimates hit the boundary.
    result = minimize_scalar(objective, bounds=(-bound, bound), method="bounded")
    at_bound = abs(abs(float(result.x)) - bound) < 1e-4
    return float(result.x), bool(result.success), int(result.nfev), at_bound


def prepare_m4_frames(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    means = {name: float(train[name].mean()) for name in ["GA", "BMI", "AGE"]}
    train_result = train.copy()
    validation_result = validation.copy()
    for name, mean in means.items():
        train_result[f"{name}_c"] = train_result[name] - mean
        validation_result[f"{name}_c"] = validation_result[name] - mean
        train_result[f"{name}_mean"] = mean
        validation_result[f"{name}_mean"] = mean
    return train_result, validation_result, means


def read_csv_if_present(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def prepare_r_runtime(r_temp: Path) -> tuple[Path, Path, dict[str, str]]:
    if not r_temp.is_absolute() or any(ord(character) > 127 for character in str(r_temp)):
        raise ValueError("--r-temp must be an absolute writable ASCII path")
    r_temp.mkdir(parents=True, exist_ok=True)
    rscript = round1.find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript not found")
    env = os.environ.copy()
    env.update(
        {"TEMP": str(r_temp), "TMP": str(r_temp), "TMPDIR": str(r_temp),
         "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C"}
    )
    probe = subprocess.run(
        [str(rscript), "--vanilla", "-e",
         "suppressPackageStartupMessages(library(mgcv)); stopifnot(exists('betar')); cat(as.character(packageVersion('mgcv')))"],
        cwd=r_temp, env=env, capture_output=True, shell=False, check=False,
    )
    if probe.returncode:
        raise RuntimeError(f"R/mgcv probe failed: {round1.decode_process(probe.stderr)}")
    runtime = r_temp / f"q1_round2_runtime_{os.getpid()}"
    runtime.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Q1_ROOT / "code" / "m4_round2.R", runtime / "m4_round2.R")
    return rscript, runtime, env


def run_m4_fit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    k_value: int,
    context: str,
    runtime: Path,
    rscript: Path,
    env: dict[str, str],
    make_grid: bool = False,
) -> dict[str, object]:
    train_r, validation_r, means = prepare_m4_frames(train, validation)
    context_dir = runtime / context
    context_dir.mkdir(parents=True, exist_ok=False)
    columns = [
        "row_id", "patient_id", "GA", "BMI", "AGE", "Y", "gravidity_cat",
        "parity", "conception_mode", "GA_c", "BMI_c", "AGE_c", "GA_mean",
        "BMI_mean", "AGE_mean",
    ]
    train_path = context_dir / "train.csv"
    validation_path = context_dir / "validation.csv"
    output_dir = context_dir / "output"
    train_r[columns].to_csv(train_path, index=False)
    validation_r[columns].to_csv(validation_path, index=False)
    process = subprocess.run(
        [str(rscript), "--vanilla", str(runtime / "m4_round2.R"), str(k_value),
         str(train_path), str(validation_path), str(output_dir), "1" if make_grid else "0"],
        cwd=runtime, env=env, capture_output=True, shell=False, check=False,
    )
    stderr = round1.decode_process(process.stderr).strip()
    error_path = output_dir / "error.txt"
    error = error_path.read_text(encoding="utf-8", errors="replace").strip() if error_path.exists() else ""
    info = round1.r_info(output_dir / "info.csv") if (output_dir / "info.csv").exists() else {}
    converged = process.returncode == 0 and str(info.get("converged", "")).upper() in {"TRUE", "1", "1.0"}
    predictions = read_csv_if_present(output_dir / "predictions.csv")
    if not predictions.empty:
        predictions = validation[["row_id", "patient_id", "fold", "GA", "BMI", "Y"]].merge(
            predictions, on="row_id", how="left", validate="one_to_one"
        )
    return {
        "context": context,
        "k": k_value,
        "means": means,
        "converged": converged,
        "warning": " | ".join(part for part in [str(info.get("warnings", "")), stderr] if part and part != "nan"),
        "error": error or (f"R exit {process.returncode}" if process.returncode else ""),
        "info": info,
        "predictions": predictions,
        "coefficients": read_csv_if_present(output_dir / "coefficients.csv"),
        "smooth_terms": read_csv_if_present(output_dir / "smooth_terms.csv"),
        "variance_components": read_csv_if_present(output_dir / "variance_components.csv"),
        "random_effect_estimates": read_csv_if_present(output_dir / "random_effect_estimates.csv"),
        "k_check": read_csv_if_present(output_dir / "k_check.csv"),
        "training_components": read_csv_if_present(output_dir / "training_components.csv"),
        "effect_grid": read_csv_if_present(output_dir / "effect_grid.csv"),
        "gam_check": (output_dir / "gam_check.txt").read_text(encoding="utf-8", errors="replace")
        if (output_dir / "gam_check.txt").exists() else "",
    }


def augment_m4_predictions(fit: dict[str, object]) -> pd.DataFrame:
    predictions = fit["predictions"].copy()
    if predictions.empty or not fit["converged"]:
        return pd.DataFrame()
    sigma_u2 = float(fit["info"].get("random_intercept_variance", np.nan))
    phi = float(fit["info"].get("precision", np.nan))
    if not np.isfinite(sigma_u2) or sigma_u2 < 0 or not np.isfinite(phi) or phi <= 0:
        return pd.DataFrame()
    marginal, pass_probability = integrate_m4(predictions["eta_fixed"].to_numpy(), sigma_u2, phi)
    predictions = predictions.rename(columns={"Y": "observed_y"})
    predictions["marginalized_population_prediction"] = marginal
    predictions["P_pass_M4P"] = pass_probability
    predictions["random_intercept_variance"] = sigma_u2
    predictions["precision_phi"] = phi
    predictions["selected_k"] = int(fit["k"])
    return predictions


def run_nested_m4(
    data: pd.DataFrame, runtime: Path, rscript: Path, env: dict[str, str], workers: int
) -> dict[str, object]:
    tuning_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    outer_predictions: list[pd.DataFrame] = []
    outer_diagnostics: list[dict[str, object]] = []
    outer_smooth: list[pd.DataFrame] = []
    outer_k_check: list[pd.DataFrame] = []
    outer_coefficients: list[pd.DataFrame] = []

    inner_jobs = []
    for outer_fold in range(1, 6):
        outer_train = data[data["fold"] != outer_fold].copy()
        inner_splitter = GroupKFold(n_splits=3)
        split_indices = list(inner_splitter.split(outer_train, groups=outer_train["patient_id"]))
        for k_value in M4_K_VALUES:
            for inner_fold, (inner_train_index, inner_validation_index) in enumerate(split_indices, 1):
                inner_jobs.append((
                    outer_fold, inner_fold, k_value,
                    outer_train.iloc[inner_train_index].copy(),
                    outer_train.iloc[inner_validation_index].copy(),
                ))

    print(f"[M4] running {len(inner_jobs)} nested fits with {workers} R workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_m4_fit, inner_train, inner_validation, k_value,
                f"outer{outer_fold}_inner{inner_fold}_k{k_value}", runtime, rscript, env,
            ): (outer_fold, inner_fold, k_value, len(inner_validation))
            for outer_fold, inner_fold, k_value, inner_train, inner_validation in inner_jobs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            outer_fold, inner_fold, k_value, n_validation = futures[future]
            fit = future.result()
            predictions = augment_m4_predictions(fit)
            row: dict[str, object] = {
                "outer_fold": outer_fold, "inner_fold": inner_fold, "k": k_value,
                "converged": bool(fit["converged"]), "warning": fit["warning"],
                "error": fit["error"], "N": n_validation,
            }
            if not predictions.empty:
                row.update(regression_metrics(
                    predictions["observed_y"], predictions["marginalized_population_prediction"]
                ))
            tuning_rows.append(row)
            if completed % 5 == 0 or completed == len(inner_jobs):
                print(f"[M4] nested fits completed: {completed}/{len(inner_jobs)}", flush=True)

    tuning = pd.DataFrame(tuning_rows).sort_values(["outer_fold", "k", "inner_fold"])
    selected_by_outer: dict[int, int] = {}
    for outer_fold in range(1, 6):
        candidate_scores: list[tuple[float, int]] = []
        outer_tuning = tuning[tuning["outer_fold"] == outer_fold]
        for k_value in M4_K_VALUES:
            candidate = outer_tuning[(outer_tuning["k"] == k_value) & outer_tuning["converged"]]
            if len(candidate) == 3 and candidate["RMSE"].notna().all():
                candidate_scores.append((float(candidate["RMSE"].mean()), k_value))
        if not candidate_scores:
            raise RuntimeError(f"all nested M4 candidates failed in outer fold {outer_fold}")
        selected_rmse, selected_k = min(candidate_scores, key=lambda item: (item[0], item[1]))
        selected_by_outer[outer_fold] = selected_k
        selection_rows.append(
            {"outer_fold": outer_fold, "selected_k": selected_k, "inner_mean_RMSE": selected_rmse}
        )
        print(f"[M4] outer fold {outer_fold}/5: selected k={selected_k}", flush=True)

    print("[M4] fitting five selected outer models", flush=True)
    outer_fits: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, 5)) as pool:
        futures = {}
        for outer_fold, selected_k in selected_by_outer.items():
            outer_train = data[data["fold"] != outer_fold].copy()
            outer_validation = data[data["fold"] == outer_fold].copy()
            future = pool.submit(
                run_m4_fit, outer_train, outer_validation, selected_k,
                f"outer{outer_fold}_selected_k{selected_k}", runtime, rscript, env,
            )
            futures[future] = outer_fold
        for future in as_completed(futures):
            outer_fits[futures[future]] = future.result()

    for outer_fold in range(1, 6):
        selected_k = selected_by_outer[outer_fold]
        outer_fit = outer_fits[outer_fold]
        predictions = augment_m4_predictions(outer_fit)
        if predictions.empty:
            raise RuntimeError(f"selected M4 fit failed in outer fold {outer_fold}: {outer_fit['error']}")
        predictions["fold"] = outer_fold
        outer_predictions.append(predictions)
        diagnostics = {
            "outer_fold": outer_fold, "selected_k": selected_k,
            "converged": outer_fit["converged"], "warning": outer_fit["warning"],
            "error": outer_fit["error"], "gam_check": outer_fit["gam_check"],
        }
        diagnostics.update(outer_fit["info"])
        outer_diagnostics.append(diagnostics)
        for key, collection in [
            ("smooth_terms", outer_smooth), ("k_check", outer_k_check),
            ("coefficients", outer_coefficients),
        ]:
            frame = outer_fit[key].copy()
            if not frame.empty:
                frame.insert(0, "outer_fold", outer_fold)
                frame.insert(1, "selected_k", selected_k)
                collection.append(frame)

    predictions = pd.concat(outer_predictions, ignore_index=True)
    predictions = predictions.sort_values("row_id").reset_index(drop=True)
    selections = pd.DataFrame(selection_rows)
    counts = selections["selected_k"].value_counts()
    most_selected = set(counts[counts == counts.max()].index.astype(int))
    inner_means = tuning[tuning["converged"]].groupby("k")["RMSE"].mean()
    final_k = min(most_selected, key=lambda k: (float(inner_means.get(k, np.inf)), k))
    print(f"[M4] full-data fit with k={final_k}", flush=True)
    final_fit = run_m4_fit(data, data, final_k, f"full_k{final_k}", runtime, rscript, env, make_grid=True)
    if not final_fit["converged"]:
        raise RuntimeError(f"final M4 fit failed: {final_fit['error']}")

    sigma_u2 = float(final_fit["info"]["random_intercept_variance"])
    phi = float(final_fit["info"]["precision"])
    components = final_fit["training_components"]
    fixed_variance = float(np.var(components["eta_fixed"], ddof=1))
    mu_conditional = np.clip(components["mu_conditional"].to_numpy(), 1e-8, 1 - 1e-8)
    residual_variance = float(np.mean(
        polygamma(1, mu_conditional * phi) + polygamma(1, (1 - mu_conditional) * phi)
    ))
    total_variance = fixed_variance + sigma_u2 + residual_variance
    structure = pd.DataFrame([{
        "selected_k": final_k,
        "fixed_effect_variance": fixed_variance,
        "random_intercept_variance": sigma_u2,
        "distribution_specific_variance": residual_variance,
        "Marginal_R2": fixed_variance / total_variance,
        "Conditional_R2": (fixed_variance + sigma_u2) / total_variance,
        "ICC": sigma_u2 / (sigma_u2 + residual_variance),
        "R2_scale": "latent logit scale; exact Beta logit variance via trigamma",
        "ICC_type": "adjusted ICC excluding fixed-effect variance",
    }])
    effect_grid = final_fit["effect_grid"].copy()
    effect_grid["population_prediction"], effect_grid["P_pass_M4P"] = integrate_m4(
        effect_grid["eta_fixed"].to_numpy(), sigma_u2, phi
    )
    return {
        "predictions": predictions,
        "fold_metrics": fold_metrics(predictions, "M4-P", "marginalized_population_prediction"),
        "u0_fold_metrics": fold_metrics(predictions, "M4-P_u0", "conditional_at_u0_prediction"),
        "tuning": tuning,
        "selections": selections,
        "outer_diagnostics": pd.DataFrame(outer_diagnostics),
        "outer_smooth": pd.concat(outer_smooth, ignore_index=True) if outer_smooth else pd.DataFrame(),
        "outer_k_check": pd.concat(outer_k_check, ignore_index=True) if outer_k_check else pd.DataFrame(),
        "outer_coefficients": pd.concat(outer_coefficients, ignore_index=True) if outer_coefficients else pd.DataFrame(),
        "final_fit": final_fit,
        "final_structure": structure,
        "effect_grid": effect_grid,
        "final_k": final_k,
    }


def sgee_design(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    ga_train = dmatrix("cr(GA, df=4) - 1", train, return_type="dataframe")
    bmi_train = dmatrix("cr(BMI, df=4) - 1", train, return_type="dataframe")
    ga_validation = np.asarray(build_design_matrices([ga_train.design_info], validation)[0])[:, 1:]
    bmi_validation = np.asarray(build_design_matrices([bmi_train.design_info], validation)[0])[:, 1:]
    # Natural-spline columns sum to one; drop one basis column when an intercept is present.
    ga_train_array = np.asarray(ga_train)[:, 1:]
    bmi_train_array = np.asarray(bmi_train)[:, 1:]
    age_mean = float(train["AGE"].mean())

    def clinical(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            frame["AGE"].to_numpy() - age_mean,
            frame["gravidity_cat"].eq("2").astype(float),
            frame["gravidity_cat"].eq("3plus").astype(float),
            frame["parity"].to_numpy(dtype=float),
            frame["conception_mode"].eq("assisted").astype(float),
        ])

    train_x = np.column_stack([
        np.ones(len(train)), ga_train_array, bmi_train_array,
        ga_train_array * bmi_train_array, clinical(train),
    ])
    validation_x = np.column_stack([
        np.ones(len(validation)), ga_validation, bmi_validation,
        ga_validation * bmi_validation, clinical(validation),
    ])
    names = (
        ["Intercept"] + [f"GA_spline_{index}" for index in range(1, 4)]
        + [f"BMI_spline_{index}" for index in range(1, 4)]
        + [f"GA_BMI_lowrank_{index}" for index in range(1, 4)]
        + ["AGE_c", "gravidity_2", "gravidity_3plus", "parity", "assisted_reproduction"]
    )
    return train_x, validation_x, names, age_mean


def fit_sgee(train: pd.DataFrame, validation: pd.DataFrame, context: str) -> dict[str, object]:
    train_x, validation_x, names, age_mean = sgee_design(train, validation)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = GEE(
            train["Y"].to_numpy(), train_x, groups=train["patient_id"].to_numpy(),
            family=Binomial(), cov_struct=Exchangeable(),
        ).fit(maxiter=200)
        predicted = np.clip(np.asarray(fit.predict(validation_x), dtype=float), 1e-8, 1 - 1e-8)
        coefficients = pd.DataFrame({
            "context": context,
            "term": names,
            "estimate": np.asarray(fit.params),
            "std_error": np.asarray(fit.bse),
            "statistic": np.asarray(fit.tvalues),
            "p_value": np.asarray(fit.pvalues),
        })
    return {
        "predicted": predicted,
        "coefficients": coefficients,
        "converged": bool(fit.converged),
        "working_correlation": float(np.asarray(fit.cov_struct.dep_params).reshape(-1)[0]),
        "warning": " | ".join(str(item.message) for item in caught),
        "age_mean": age_mean,
    }


def run_sgee(data: pd.DataFrame) -> dict[str, object]:
    predictions, coefficients, diagnostics = [], [], []
    for fold in range(1, 6):
        train = data[data["fold"] != fold]
        validation = data[data["fold"] == fold]
        fit = fit_sgee(train, validation, f"fold_{fold}")
        frame = validation[["row_id", "patient_id", "fold", "GA", "BMI", "Y"]].copy()
        frame = frame.rename(columns={"Y": "observed_y"})
        frame["prediction"] = fit["predicted"]
        predictions.append(frame)
        coefficients.append(fit["coefficients"])
        diagnostics.append({
            "context": f"fold_{fold}", "converged": fit["converged"],
            "working_correlation": fit["working_correlation"], "warning": fit["warning"],
            "AGE_train_mean": fit["age_mean"],
        })
    full = fit_sgee(data, data, "full")
    coefficients.append(full["coefficients"])
    diagnostics.append({
        "context": "full", "converged": full["converged"],
        "working_correlation": full["working_correlation"], "warning": full["warning"],
        "AGE_train_mean": full["age_mean"],
    })
    prediction_frame = pd.concat(predictions, ignore_index=True).sort_values("row_id")
    return {
        "predictions": prediction_frame,
        "fold_metrics": fold_metrics(prediction_frame, "SGEE", "prediction"),
        "coefficients": pd.concat(coefficients, ignore_index=True),
        "diagnostics": pd.DataFrame(diagnostics),
        "converged": all(row["converged"] for row in diagnostics),
    }


def run_b0(data: pd.DataFrame, manifest: dict[str, object]) -> dict[str, object]:
    result = round1.run_b0(data, manifest)
    predictions = result["cv_predictions"].rename(
        columns={"observed": "observed_y", "predicted": "prediction"}
    )
    predictions = predictions.merge(
        data[["row_id", "GA", "BMI"]], on="row_id", how="left", validate="one_to_one"
    )
    first_round = pd.read_csv(Q1_ROOT / "outputs" / "decision" / "model_decision_table.csv")
    expected_rmse = float(first_round.set_index("Model").loc["B0", "CV_RMSE"])
    actual_rmse = float(result["cv_fold_metrics"]["RMSE"].mean())
    if not math.isclose(expected_rmse, actual_rmse, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("round-two B0 no longer reproduces the frozen first-round benchmark")
    return {
        "predictions": predictions,
        "fold_metrics": fold_metrics(predictions, "B0", "prediction"),
        "coefficients": result["coefficients"],
        "diagnostics": result["diagnostics"],
        "convergence": result["convergence"],
    }


def build_seen_predictions(
    data: pd.DataFrame, b0_predictions: pd.DataFrame, m4_predictions: pd.DataFrame
) -> pd.DataFrame:
    b0_lookup = b0_predictions.set_index("row_id")["prediction"]
    m4_lookup = m4_predictions.set_index("row_id")
    rows = []
    for patient_id, patient in data.groupby("patient_id", sort=True):
        patient = patient.sort_values(["GA", "blood_draw_no", "row_id"]).reset_index(drop=True)
        if len(patient) < 2:
            continue
        for target_position in range(1, len(patient)):
            history = patient.iloc[:target_position]
            target = patient.iloc[target_position]
            history_m4 = m4_lookup.loc[history["row_id"]]
            target_m4 = m4_lookup.loc[target["row_id"]]
            u_hat, success, nfev, at_bound = estimate_random_intercept(
                history["Y"].to_numpy(), history_m4["eta_fixed"].to_numpy(),
                float(target_m4["precision_phi"]), float(target_m4["random_intercept_variance"]),
            )
            history_n = len(history)
            history_group = "1" if history_n == 1 else ("2" if history_n == 2 else ">=3")
            rows.append({
                "fold": int(target["fold"]), "patient_id": patient_id,
                "target_row_id": target["row_id"], "target_GA": float(target["GA"]),
                "observed_y": float(target["Y"]), "history_n": history_n,
                "history_group": history_group, "history_row_ids": ";".join(history["row_id"]),
                "history_last_GA": float(history["GA"].iloc[-1]),
                "B0_prediction": float(b0_lookup.loc[target["row_id"]]),
                "M4_population_prediction": float(target_m4["marginalized_population_prediction"]),
                "M4_conditional_prediction": float(expit(float(target_m4["eta_fixed"]) + u_hat)),
                "random_effect_estimate": u_hat, "map_success": success,
                "map_nfev": nfev, "map_at_bound": at_bound,
                "selected_k": int(target_m4["selected_k"]),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or not frame["map_success"].all() or frame["map_at_bound"].any():
        raise RuntimeError("one-step-ahead empirical-Bayes update failed or hit its search bound")
    return frame


def seen_metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = {
        "B0": "B0_prediction",
        "M4-P": "M4_population_prediction",
        "M4-C": "M4_conditional_prediction",
    }
    overall_rows, history_rows = [], []
    for model, column in mapping.items():
        overall_rows.append({
            "Model": model, "N": len(predictions),
            **regression_metrics(predictions["observed_y"], predictions[column]),
            "Uses_History": model == "M4-C",
        })
        for history_group, frame in predictions.groupby("history_group", sort=False):
            history_rows.append({
                "history_n": history_group, "Model": model, "N": len(frame),
                **regression_metrics(frame["observed_y"], frame[column]),
            })
    overall = pd.DataFrame(overall_rows)
    history = pd.DataFrame(history_rows)
    gain_rows = []
    for history_group in ["1", "2", ">=3"]:
        subset = history[history["history_n"] == history_group].set_index("Model")
        population_rmse = float(subset.loc["M4-P", "RMSE"])
        conditional_rmse = float(subset.loc["M4-C", "RMSE"])
        gain_rows.append({
            "history_n": history_group,
            "N": int(subset.loc["M4-C", "N"]),
            "M4-P_RMSE": population_rmse,
            "M4-C_RMSE": conditional_rmse,
            "Absolute_improvement": population_rmse - conditional_rmse,
            "Relative_improvement_pct": 100 * (population_rmse - conditional_rmse) / population_rmse,
        })
    return overall, history, pd.DataFrame(gain_rows)


def probability_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    truth = predictions["observed_y"].ge(THRESHOLD).astype(int).to_numpy()
    probability = predictions["P_pass_M4P"].clip(1e-8, 1 - 1e-8).to_numpy()
    calibration = sm.GLM(truth, sm.add_constant(logit(probability)), family=sm.families.Binomial()).fit()
    return pd.DataFrame([{
        "Brier_Score": float(np.mean((truth - probability) ** 2)),
        "Calibration_intercept": float(calibration.params[0]),
        "Calibration_slope": float(calibration.params[1]),
        "Probability_AUC": float(roc_auc_score(truth, probability)),
        "N": len(truth),
    }])


def unseen_decision(
    b0: dict[str, object], m4: dict[str, object], sgee: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_inputs = {
        "B0": (b0["predictions"], b0["fold_metrics"], "prediction", True, "Low"),
        "M4-P": (m4["predictions"], m4["fold_metrics"], "marginalized_population_prediction", True, "High"),
        "SGEE": (sgee["predictions"], sgee["fold_metrics"], "prediction", sgee["converged"], "Medium"),
    }
    rows, threshold_rows = [], []
    for model, (predictions, metrics, prediction_column, stable, complexity) in model_inputs.items():
        boundary = predictions[predictions["observed_y"].between(BOUNDARY_LOW, BOUNDARY_HIGH)]
        threshold = threshold_metrics(predictions["observed_y"], predictions[prediction_column])
        threshold_rows.append({"Model": model, **threshold})
        rows.append({
            "Model": model,
            "CV_RMSE": float(metrics["RMSE"].mean()),
            "RMSE_SD": float(metrics["RMSE"].std(ddof=1)),
            "CV_MAE": float(metrics["MAE"].mean()),
            "MAE_SD": float(metrics["MAE"].std(ddof=1)),
            "CV_R2": float(metrics["R2"].mean()),
            "R2_SD": float(metrics["R2"].std(ddof=1)),
            "Boundary_RMSE": regression_metrics(boundary["observed_y"], boundary[prediction_column])["RMSE"],
            "Boundary_N": len(boundary),
            "Pass_Accuracy": threshold["Accuracy"], "Pass_Precision": threshold["Precision"],
            "Pass_Recall": threshold["Recall"], "Pass_F1": threshold["F1"],
            "Pass_AUC": threshold["AUC"], "Stable": bool(stable),
            "Complexity": complexity,
        })
    table = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
    best_model = str(table.iloc[0]["Model"])
    table["Decision"] = np.where(table["Model"].eq(best_model), "推荐", "保留对照")
    return table, pd.DataFrame(threshold_rows)


def create_figures(
    unseen: pd.DataFrame,
    m4_predictions: pd.DataFrame,
    seen_predictions: pd.DataFrame,
    gain: pd.DataFrame,
    effect_grid: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = {"B0": "#0072B2", "M4-P": "#D55E00", "SGEE": "#009E73", "M4-C": "#CC79A7"}

    order = ["B0", "M4-P", "SGEE"]
    plot_table = unseen.set_index("Model").loc[order]
    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - width / 2, plot_table["CV_RMSE"], width, yerr=plot_table["RMSE_SD"],
           capsize=3, color=[colors[item] for item in order], edgecolor="black", linewidth=0.5, label="RMSE")
    ax.bar(x + width / 2, plot_table["CV_MAE"], width, yerr=plot_table["MAE_SD"],
           capsize=3, color="white", edgecolor=[colors[item] for item in order], linewidth=1.5, hatch="//", label="MAE")
    ax.set_xticks(x, order)
    ax.set_ylabel("Error on raw Y scale")
    ax.set_title("Unseen patients: five-fold Group-CV (error bars: fold SD)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "unseen_model_comparison.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True)
    limits = [0, max(0.24, seen_predictions["observed_y"].max() * 1.03)]
    for axis, column, label, color in [
        (axes[0], "M4_population_prediction", "M4-P", colors["M4-P"]),
        (axes[1], "M4_conditional_prediction", "M4-C", colors["M4-C"]),
    ]:
        axis.scatter(seen_predictions["observed_y"], seen_predictions[column], s=11, alpha=0.48,
                     color=color, edgecolor="none")
        axis.plot(limits, limits, color="black", linestyle="--", linewidth=0.9)
        axis.set_title(label)
        axis.set_xlabel("Observed Y")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Out-of-fold one-step prediction")
    axes[0].set_xlim(limits); axes[0].set_ylim(limits)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "m4_population_vs_conditional.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    x = np.arange(3)
    ax.plot(x, gain["M4-P_RMSE"], color=colors["M4-P"], marker="o", linestyle="--", label="M4-P")
    ax.plot(x, gain["M4-C_RMSE"], color=colors["M4-C"], marker="s", linestyle="-", label="M4-C")
    ax.set_xticks(x, [f"{group}\n(n={count})" for group, count in zip(gain["history_n"], gain["N"])])
    ax.set_xlabel("Number of prior tests")
    ax.set_ylabel("One-step-ahead RMSE")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "personalization_gain.png", dpi=300)
    plt.close(fig)

    pivot = effect_grid.pivot(index="BMI", columns="GA", values="population_prediction")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    contour = ax.contourf(pivot.columns, pivot.index, pivot.to_numpy(), levels=20, cmap="viridis")
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Population-marginalized predicted Y")
    ax.set_xlabel("Gestational age (weeks)")
    ax.set_ylabel("BMI (kg/m²)")
    ax.set_title("M4 population-level response surface")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "m4_effect_surface.png", dpi=300)
    plt.close(fig)


def clean_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return None
    return value if isinstance(value, (str, int, float)) else str(value)


def sheet_payload(name: str, frame: pd.DataFrame) -> dict[str, object]:
    clean = frame.copy().replace([np.inf, -np.inf], np.nan)
    rows = [[clean_value(column) for column in clean.columns]]
    rows.extend([[clean_value(value) for value in row] for row in clean.itertuples(index=False, name=None)])
    return {"name": name[:31], "rows": rows}


def info_frame(values: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{"key": key, "value": clean_value(value)} for key, value in values.items()])


def write_artifact_payload(workbooks: list[dict[str, object]]) -> None:
    ROUND2_DIR.mkdir(parents=True, exist_ok=True)
    PAYLOAD_PATH.write_text(json.dumps({"workbooks": workbooks}, ensure_ascii=False), encoding="utf-8")


def build_workbook_specs(
    manifest: dict[str, object], b0: dict[str, object], m4: dict[str, object], sgee: dict[str, object],
    unseen: pd.DataFrame, threshold_table: pd.DataFrame, probability: pd.DataFrame,
    seen_predictions: pd.DataFrame, seen_overall: pd.DataFrame, seen_history: pd.DataFrame,
    gain: pd.DataFrame,
) -> list[dict[str, object]]:
    def book(relative_path: str, sheets: dict[str, pd.DataFrame]) -> dict[str, object]:
        return {"path": relative_path, "sheets": [sheet_payload(name, frame) for name, frame in sheets.items()]}

    m4_final = m4["final_fit"]
    outer_diagnostics = m4["outer_diagnostics"].drop(columns=["gam_check"], errors="ignore")
    gam_check_rows = []
    if "gam_check" in m4["outer_diagnostics"]:
        for _, row in m4["outer_diagnostics"].iterrows():
            for line_number, line in enumerate(str(row["gam_check"]).splitlines(), 1):
                gam_check_rows.append({
                    "outer_fold": row["outer_fold"], "selected_k": row["selected_k"],
                    "line_number": line_number, "gam_check_line": line,
                })
    outer_gam_check = pd.DataFrame(gam_check_rows)
    final_gam_check = pd.DataFrame({
        "line_number": np.arange(1, len(str(m4_final["gam_check"]).splitlines()) + 1),
        "gam_check_line": str(m4_final["gam_check"]).splitlines(),
    })
    model_info = info_frame({
        "source_sha256": manifest["source_sha256"], "seed": SEED,
        "outer_cv": "reused 5-fold GroupKFold by patient", "inner_cv": "3-fold GroupKFold by patient",
        "quadrature_nodes": QUADRATURE_NODES, "threshold": THRESHOLD,
        "clinical_covariates": "gravidity category; parity count; natural vs assisted conception",
    })
    unseen_combined = pd.concat([
        b0["predictions"].assign(Model="B0").rename(columns={"prediction": "predicted_y"}),
        m4["predictions"].assign(Model="M4-P").rename(columns={"marginalized_population_prediction": "predicted_y"}),
        sgee["predictions"].assign(Model="SGEE").rename(columns={"prediction": "predicted_y"}),
    ], ignore_index=True, sort=False)
    all_fold_metrics = pd.concat([b0["fold_metrics"], m4["fold_metrics"], sgee["fold_metrics"]], ignore_index=True)
    return [
        book("raw/unseen_B0_raw.xlsx", {
            "model_info": model_info, "cv_fold_metrics": b0["fold_metrics"],
            "cv_predictions": b0["predictions"], "coefficients": b0["coefficients"],
            "diagnostics": b0["diagnostics"], "convergence": b0["convergence"],
            "threshold_metrics": threshold_table[threshold_table["Model"] == "B0"],
        }),
        book("raw/unseen_M4P_raw.xlsx", {
            "model_info": model_info, "cv_fold_metrics": m4["fold_metrics"],
            "u0_fold_metrics": m4["u0_fold_metrics"], "cv_predictions": m4["predictions"],
            "outer_diagnostics": outer_diagnostics, "outer_gam_check": outer_gam_check,
            "outer_smooth_terms": m4["outer_smooth"],
            "outer_k_check": m4["outer_k_check"], "outer_coefficients": m4["outer_coefficients"],
            "probability_calibration": probability,
        }),
        book("raw/unseen_SGEE_raw.xlsx", {
            "model_info": model_info, "cv_fold_metrics": sgee["fold_metrics"],
            "cv_predictions": sgee["predictions"], "coefficients": sgee["coefficients"],
            "diagnostics": sgee["diagnostics"],
            "threshold_metrics": threshold_table[threshold_table["Model"] == "SGEE"],
        }),
        book("raw/seen_M4C_raw.xlsx", {
            "model_info": model_info, "overall_metrics": seen_overall,
            "history_metrics": seen_history, "personalization_gain": gain,
            "one_step_predictions": seen_predictions,
        }),
        book("raw/m4_tuning_raw.xlsx", {
            "inner_metrics": m4["tuning"], "outer_selection": m4["selections"],
            "outer_diagnostics": outer_diagnostics, "outer_gam_check": outer_gam_check,
            "outer_smooth_terms": m4["outer_smooth"],
            "outer_k_check": m4["outer_k_check"], "final_info": info_frame(m4_final["info"]),
            "final_coefficients": m4_final["coefficients"], "final_smooth_terms": m4_final["smooth_terms"],
            "final_k_check": m4_final["k_check"], "final_gam_check": final_gam_check,
            "final_variance": m4_final["variance_components"],
            "final_random_effects": m4_final["random_effect_estimates"],
            "m4_R2_structure": m4["final_structure"], "effect_surface": m4["effect_grid"],
        }),
        book("raw/round2_all_raw.xlsx", {
            "model_info": model_info, "unseen_decision": unseen,
            "unseen_fold_metrics": all_fold_metrics, "unseen_predictions": unseen_combined,
            "seen_decision": seen_overall, "seen_history_metrics": seen_history,
            "seen_predictions": seen_predictions, "personalization_gain": gain,
            "m4_inner_tuning": m4["tuning"], "m4_outer_selection": m4["selections"],
            "m4_structure": m4["final_structure"], "m4_probability": probability,
        }),
        book("decision/unseen_decision_table.xlsx", {"unseen_decision": unseen}),
        book("decision/seen_decision_table.xlsx", {"seen_decision": seen_overall}),
        book("decision/personalization_gain.xlsx", {"personalization_gain": gain}),
    ]


def write_summary(
    unseen: pd.DataFrame, m4: dict[str, object], sgee: dict[str, object],
    seen_overall: pd.DataFrame, gain: pd.DataFrame,
) -> None:
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    unseen_rank = unseen.sort_values("CV_RMSE").reset_index(drop=True)
    b0_row = unseen.set_index("Model").loc["B0"]
    m4_row = unseen.set_index("Model").loc["M4-P"]
    sgee_row = unseen.set_index("Model").loc["SGEE"]
    u0_rmse = float(m4["u0_fold_metrics"]["RMSE"].mean())
    marginal_rmse = float(m4["fold_metrics"]["RMSE"].mean())
    seen_index = seen_overall.set_index("Model")
    m4c_rmse = float(seen_index.loc["M4-C", "RMSE"])
    m4p_seen_rmse = float(seen_index.loc["M4-P", "RMSE"])
    b0_seen_rmse = float(seen_index.loc["B0", "RMSE"])
    best_unseen = str(unseen_rank.iloc[0]["Model"])
    best_seen = str(seen_overall.sort_values("RMSE").iloc[0]["Model"])
    structure = m4["final_structure"].iloc[0]
    m4_population_acceptable = (
        float(m4_row["CV_RMSE"]) <= 1.02 * float(b0_row["CV_RMSE"])
        and float(m4_row["Boundary_RMSE"]) < float(b0_row["Boundary_RMSE"])
        and float(m4_row["Pass_F1"]) >= float(b0_row["Pass_F1"])
    )
    unified = best_seen == "M4-C" and (best_unseen == "M4-P" or m4_population_acceptable)
    framework_note = (
        "可采用同一M4模型完成群体到个体化过渡。"
        if unified else "更合适的是B0群体预测→M4-C个体化预测的双场景框架。"
    )

    lines = [
        "# Q1 第二轮模型优化与验证结论", "", "## 1. 新孕妇结果", "",
        "| Rank | Model | RMSE | MAE | R2 | Boundary_RMSE | 状态 |",
        "|---:|:---|---:|---:|---:|---:|:---|",
    ]
    for rank, row in unseen_rank.iterrows():
        lines.append(
            f"| {rank + 1} | {row['Model']} | {row['CV_RMSE']:.6f} | {row['CV_MAE']:.6f} | "
            f"{row['CV_R2']:.6f} | {row['Boundary_RMSE']:.6f} | {'稳定' if row['Stable'] else '不稳定'} |"
        )
    marginal_change = 100 * (u0_rmse - marginal_rmse) / u0_rmse
    lines += [
        "",
        f"- B0 是否仍然最好：{'是' if best_unseen == 'B0' else '否，' + best_unseen + ' 更好'}。",
        f"- M4-P 边际化后是否改善：相对同一外层模型的 u=0 预测，RMSE变化 {marginal_change:+.2f}%（正值为改善）。",
        f"- SGEE 是否提供额外优势：{'是' if float(sgee_row['CV_RMSE']) < min(float(b0_row['CV_RMSE']), float(m4_row['CV_RMSE'])) else '否'}。",
        "", "## 2. 已见孕妇结果", "",
        "| Model | RMSE | MAE | R2 |", "|:---|---:|---:|---:|",
    ]
    for _, row in seen_overall.iterrows():
        lines.append(f"| {row['Model']} | {row['RMSE']:.6f} | {row['MAE']:.6f} | {row['R2']:.6f} |")
    lines += [
        "",
        f"- M4-C 是否明显超过 M4-P：{'是' if m4c_rmse <= 0.98 * m4p_seen_rmse else '否'}。",
    ]
    for _, row in gain.iterrows():
        lines.append(
            f"- 有 {row['history_n']} 次历史检测：M4-C 相对 M4-P 的 RMSE 改善 "
            f"{row['Absolute_improvement']:.6f}（{row['Relative_improvement_pct']:.2f}%）。"
        )
    lines += [
        "", "## 3. M4 结构解释", "",
        f"- ICC：{structure['ICC']:.6f}",
        f"- Marginal R²：{structure['Marginal_R2']:.6f}",
        f"- Conditional R²：{structure['Conditional_R2']:.6f}",
        "- 高拟合度主要来源：" + ("孕妇个体随机效应。" if structure["Conditional_R2"] - structure["Marginal_R2"] > structure["Marginal_R2"] else "孕周/BMI等固定效应。"),
        "", "## 4. 最终建议", "",
        f"- 新孕妇推荐模型：{best_unseen}",
        f"- 已有历史孕妇推荐模型：{best_seen}",
        f"- 是否建议最终论文采用统一 M4 population→personalized 框架：{'是' if unified else '否'}",
        (
            f"- 主要理由：新孕妇 M4-P 相对 B0 的 RMSE差异为 "
            f"{100 * (float(m4_row['CV_RMSE']) / float(b0_row['CV_RMSE']) - 1):+.2f}%；"
            f"已有历史时 M4-C 相对 M4-P 的总体 RMSE差异为 "
            f"{100 * (m4c_rmse / m4p_seen_rmse - 1):+.2f}%，"
            f"相对 B0 为 {100 * (m4c_rmse / b0_seen_rmse - 1):+.2f}%。{framework_note}"
        ),
    ]
    (DECISION_DIR / "decision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1 round-two population-to-personalized validation")
    parser.add_argument("--r-temp", type=Path, help="absolute writable ASCII directory for R TEMP/TMP")
    parser.add_argument("--r-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    if args.r_temp is None:
        parser.error("--r-temp is required")
    if args.r_workers < 1:
        parser.error("--r-workers must be positive")

    np.random.seed(SEED)
    for directory in [RAW_DIR, DECISION_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    data, manifest = load_round2_data()
    rscript, runtime, env = prepare_r_runtime(args.r_temp.resolve())
    print("[1/4] frozen B0 benchmark", flush=True)
    b0 = run_b0(data, manifest)
    print("[2/4] nested M4-P and final M4", flush=True)
    m4 = run_nested_m4(data, runtime, rscript, env, args.r_workers)
    print("[3/4] Spline-GEE and one-step personalization", flush=True)
    sgee = run_sgee(data)
    seen_predictions = build_seen_predictions(data, b0["predictions"], m4["predictions"])
    seen_overall, seen_history, gain = seen_metric_tables(seen_predictions)
    unseen, threshold_table = unseen_decision(b0, m4, sgee)
    probability = probability_calibration(m4["predictions"])
    seen_overall["Decision"] = np.where(
        seen_overall["RMSE"].eq(seen_overall["RMSE"].min()), "推荐", "保留对照"
    )
    print("[4/4] figures, decision summary, and workbook payload", flush=True)
    create_figures(unseen, m4["predictions"], seen_predictions, gain, m4["effect_grid"])
    write_summary(unseen, m4, sgee, seen_overall, gain)
    workbooks = build_workbook_specs(
        manifest, b0, m4, sgee, unseen, threshold_table, probability,
        seen_predictions, seen_overall, seen_history, gain,
    )
    write_artifact_payload(workbooks)
    unseen.to_csv(DECISION_DIR / "unseen_decision_table.csv", index=False, encoding="utf-8-sig")
    seen_overall.to_csv(DECISION_DIR / "seen_decision_table.csv", index=False, encoding="utf-8-sig")
    gain.to_csv(DECISION_DIR / "personalization_gain.csv", index=False, encoding="utf-8-sig")
    print(f"model outputs complete; workbook payload: {PAYLOAD_PATH}", flush=True)


if __name__ == "__main__":
    main()
