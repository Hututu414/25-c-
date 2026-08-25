from __future__ import annotations

import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from lightgbm import LGBMClassifier
from prepare_q4_data import prepare_data
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

SEED = 20260824
Q4_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = Q4_ROOT / "code"
DATA_DIR = Q4_ROOT / "data_processed"
OUTPUT_DIR = Q4_ROOT / "outputs"
RAW_DIR = OUTPUT_DIR / "raw"
DECISION_DIR = OUTPUT_DIR / "decision"
FIGURE_DIR = OUTPUT_DIR / "figures"
TMP_ROOT = Q4_ROOT / ".project-tmp"
CALIBRATION_CACHE = DATA_DIR / ".calibration_cache"
PAYLOAD_PATH = OUTPUT_DIR / ".q4_workbook_payload.json"
R_SCRIPT = CODE_DIR / "calibrate_z.R"
CHROMOSOMES = (13, 18, 21)
MODEL_KEYS = ("B0", "B1", "B1_FDR", "M1_raw", "M1", "M2", "M3")
DISPLAY_NAME = {
    "B0": "B0 raw Z>3",
    "B1": "B1 calibrated Z*>3",
    "B1_FDR": "B1 calibrated BH-FDR",
    "M1_raw": "M1-raw Elastic-Net",
    "M1": "M1 calibrated Elastic-Net",
    "M2": "M2 calibrated Random Forest",
    "M3": "M3 calibrated LightGBM",
}
QC_COLUMNS = (
    "raw_reads", "unique_reads", "alignment_rate", "duplication_rate", "filtered_rate",
    "GC_global", "GC13", "GC18", "GC21",
)
QC_RULES = {
    "unique_reads": ("low", -4.0),
    "alignment_rate": ("low", -4.0),
    "filtered_rate": ("high", 4.0),
    "GC_global": ("absolute", 4.0),
    "GC13": ("absolute", 4.0),
    "GC18": ("absolute", 4.0),
    "GC21": ("absolute", 4.0),
}
COMMON_FEATURES = [
    "X_Z", "X_conc", "GC13", "GC18", "GC21", "GC_global",
    "log_raw_reads", "log_unique_reads", "alignment_rate", "duplication_rate",
    "filtered_rate", "BMI", "GA", "age",
]


def patient_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("patient_id")["patient_id"].transform("size")
    return (1.0 / counts).to_numpy(float)


def robust_qc_fit(train: pd.DataFrame, fold: int) -> tuple[dict[str, tuple[float, float]], list[dict[str, object]]]:
    fitted: dict[str, tuple[float, float]] = {}
    rows: list[dict[str, object]] = []
    for column in QC_COLUMNS:
        values = train[column].to_numpy(float)
        location = float(np.median(values))
        scale = float(1.4826 * np.median(np.abs(values - location)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(values, ddof=1))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        fitted[column] = (location, scale)
        direction, cutoff = QC_RULES.get(column, ("audit_only", np.nan))
        rows.append(
            {
                "fold": fold, "metric": column, "robust_location": location,
                "robust_scale": scale, "rule": direction, "robust_z_cutoff": cutoff,
                "estimated_from": "outer training records only",
            }
        )
    return fitted, rows


def robust_qc_apply(frame: pd.DataFrame, fitted: dict[str, tuple[float, float]]) -> tuple[np.ndarray, list[str]]:
    flags = np.zeros(len(frame), dtype=bool)
    reasons = [[] for _ in range(len(frame))]
    for column, (direction, cutoff) in QC_RULES.items():
        location, scale = fitted[column]
        z = (frame[column].to_numpy(float) - location) / scale
        if direction == "low":
            current = z < cutoff
        elif direction == "high":
            current = z > cutoff
        else:
            current = np.abs(z) > cutoff
        flags |= current
        for index in np.flatnonzero(current):
            reasons[index].append(column)
    return flags, [";".join(items) for items in reasons]


def rscript_path() -> str:
    resolved = shutil.which("Rscript.exe") or shutil.which("Rscript")
    fallback = Path(r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe")
    if resolved:
        return resolved
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("Rscript is required for fold-wise mgcv calibration")


def calibrate_partition(
    train: pd.DataFrame, apply_data: pd.DataFrame, fold_tag: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = CALIBRATION_CACHE / fold_tag
    cached_paths = {name: cache_dir / f"{name}.csv" for name in ("train_out", "apply_out", "diagnostics")}
    if all(path.exists() for path in cached_paths.values()):
        train_result = pd.read_csv(cached_paths["train_out"])
        apply_result = pd.read_csv(cached_paths["apply_out"])
        train_result.index = train.index
        apply_result.index = apply_data.index
        return train_result, apply_result, pd.read_csv(cached_paths["diagnostics"])
    task_dir = Path(tempfile.mkdtemp(prefix=f"cal_{fold_tag}_", dir=TMP_ROOT))
    train_copy, apply_copy = train.copy(), apply_data.copy()
    calibration_covariates = ["GA", "BMI", "X_Z", "GC13", "GC18", "GC21", "GC_global", "alignment_rate", "duplication_rate", "log_unique_reads", "filtered_rate"]
    for column in calibration_covariates:
        median = float(train_copy[column].median())
        train_copy[column] = train_copy[column].fillna(median)
        apply_copy[column] = apply_copy[column].fillna(median)
    paths = {name: task_dir / f"{name}.csv" for name in ("train", "apply", "train_out", "apply_out", "diagnostics")}
    train_copy.to_csv(paths["train"], index=False, encoding="utf-8")
    apply_copy.to_csv(paths["apply"], index=False, encoding="utf-8")
    env = os.environ.copy()
    env.update({"TEMP": str(TMP_ROOT.resolve()), "TMP": str(TMP_ROOT.resolve()), "TMPDIR": str(TMP_ROOT.resolve())})
    relative = {name: path.relative_to(Q4_ROOT).as_posix() for name, path in paths.items()}
    command = [
        rscript_path(), "--vanilla", "code/calibrate_z.R", relative["train"], relative["apply"],
        relative["train_out"], relative["apply_out"], relative["diagnostics"], fold_tag,
    ]
    completed = subprocess.run(command, cwd=Q4_ROOT, env=env, capture_output=True, shell=False, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        stdout = completed.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"R calibration failed ({fold_tag}):\n{stdout}\n{stderr}")
    train_result = pd.read_csv(paths["train_out"])
    apply_result = pd.read_csv(paths["apply_out"])
    train_result.index = train.index
    apply_result.index = apply_data.index
    diagnostics = pd.read_csv(paths["diagnostics"])
    for chromosome in CHROMOSOMES:
        for result in (train_result, apply_result):
            result[f"abs_Z{chromosome}_cal"] = result[f"Z{chromosome}_cal"].abs()
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_result.to_csv(cached_paths["train_out"], index=False, encoding="utf-8")
    apply_result.to_csv(cached_paths["apply_out"], index=False, encoding="utf-8")
    diagnostics.to_csv(cached_paths["diagnostics"], index=False, encoding="utf-8")
    shutil.rmtree(task_dir)
    return train_result, apply_result, diagnostics


def raw_features() -> list[str]:
    return ["Z13", "Z18", "Z21", "abs_Z13", "abs_Z18", "abs_Z21", *COMMON_FEATURES]


def calibrated_features() -> list[str]:
    return [
        "Z13_cal", "Z18_cal", "Z21_cal", "abs_Z13_cal", "abs_Z18_cal", "abs_Z21_cal",
        *COMMON_FEATURES,
    ]


def add_raw_absolute(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for chromosome in CHROMOSOMES:
        result[f"abs_Z{chromosome}"] = result[f"Z{chromosome}"].abs()
    return result


def parameter_grid(model_key: str) -> list[dict[str, object]]:
    if model_key in {"M1", "M1_raw"}:
        return [
            {"C": 0.2, "l1_ratio": 0.2}, {"C": 1.0, "l1_ratio": 0.2},
            {"C": 1.0, "l1_ratio": 0.8}, {"C": 5.0, "l1_ratio": 0.5},
        ]
    if model_key == "M2":
        return [
            {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 5, "max_features": "sqrt"},
            {"n_estimators": 400, "max_depth": 6, "min_samples_leaf": 4, "max_features": "sqrt"},
            {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 3, "max_features": 0.7},
        ]
    if model_key == "M3":
        return [
            {"num_leaves": 7, "max_depth": 3, "learning_rate": 0.03, "min_child_samples": 15, "feature_fraction": 0.8},
            {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.03, "min_child_samples": 10, "feature_fraction": 0.9},
            {"num_leaves": 15, "max_depth": 5, "learning_rate": 0.05, "min_child_samples": 10, "feature_fraction": 0.8},
        ]
    raise KeyError(model_key)


def make_estimator(model_key: str, parameters: dict[str, object], y: np.ndarray) -> Pipeline:
    if model_key in {"M1", "M1_raw"}:
        estimator: BaseEstimator = LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced", max_iter=5000,
            random_state=SEED, n_jobs=1, **parameters,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)]
    elif model_key == "M2":
        estimator = RandomForestClassifier(
            class_weight="balanced_subsample", random_state=SEED, n_jobs=1, **parameters,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    elif model_key == "M3":
        positives = max(int(y.sum()), 1)
        estimator = LGBMClassifier(
            objective="binary", n_estimators=250, scale_pos_weight=(len(y) - positives) / positives,
            random_state=SEED, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True,
            subsample=1.0, colsample_bytree=float(parameters["feature_fraction"]),
            num_leaves=int(parameters["num_leaves"]), max_depth=int(parameters["max_depth"]),
            learning_rate=float(parameters["learning_rate"]), min_child_samples=int(parameters["min_child_samples"]),
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    else:
        raise KeyError(model_key)
    return Pipeline(steps)


def minimax_threshold(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    unique = np.unique(np.clip(probability, 0, 1))
    thresholds = np.unique(np.r_[0.0, 1.0, unique, (unique[:-1] + unique[1:]) / 2])
    best: tuple[float, float, float, float, float] | None = None
    for threshold in thresholds:
        prediction = probability >= threshold
        positive_weight = weights[y == 1].sum()
        negative_weight = weights[y == 0].sum()
        fnr = float(weights[(y == 1) & (~prediction)].sum() / positive_weight)
        fpr = float(weights[(y == 0) & prediction].sum() / negative_weight)
        candidate = (max(fnr, fpr), fnr + fpr, abs(float(threshold) - 0.5), float(threshold), fnr, fpr)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    return best[3], best[4], best[5], best[0]


def tune_label_model(
    model_key: str,
    chromosome: int,
    outer_train: pd.DataFrame,
    inner_sets: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[dict[str, object], np.ndarray, float, float, float, float]:
    label = f"true_T{chromosome}"
    features = raw_features() if model_key == "M1_raw" else calibrated_features()
    weights = patient_weights(outer_train)
    best_parameters: dict[str, object] | None = None
    best_probability: np.ndarray | None = None
    best_ap = -math.inf
    for parameters in parameter_grid(model_key):
        oof = np.full(len(outer_train), np.nan)
        for inner_train, inner_validation in inner_sets:
            y_train = inner_train[label].to_numpy(int)
            if len(np.unique(y_train)) != 2:
                raise ValueError(f"inner training split lacks both classes: {model_key} {label}")
            estimator = make_estimator(model_key, parameters, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator.fit(inner_train[features], y_train)
            positions = outer_train.index.get_indexer(inner_validation.index)
            oof[positions] = estimator.predict_proba(inner_validation[features])[:, 1]
        if np.isnan(oof).any():
            raise AssertionError("inner OOF probabilities are incomplete")
        score = float(average_precision_score(outer_train[label], oof, sample_weight=weights))
        if score > best_ap + 1e-12:
            best_ap, best_parameters, best_probability = score, parameters, oof
    assert best_parameters is not None and best_probability is not None
    threshold, fnr, fpr, risk = minimax_threshold(
        outer_train[label].to_numpy(int), best_probability, weights
    )
    return best_parameters, best_probability, threshold, fnr, fpr, risk


def metric_record(
    y: np.ndarray, probability: np.ndarray, prediction: np.ndarray, weights: np.ndarray | None
) -> dict[str, float]:
    kwargs = {} if weights is None else {"sample_weight": weights}
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1], **kwargs).ravel()
    return {
        "PR-AUC": float(average_precision_score(y, probability, **kwargs)),
        "ROC-AUC": float(roc_auc_score(y, probability, **kwargs)) if len(np.unique(y)) == 2 else np.nan,
        "Recall": float(recall_score(y, prediction, zero_division=0, **kwargs)),
        "Precision": float(precision_score(y, prediction, zero_division=0, **kwargs)),
        "F1": float(f1_score(y, prediction, zero_division=0, **kwargs)),
        "F2": float(fbeta_score(y, prediction, beta=2, zero_division=0, **kwargs)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "Balanced Accuracy": float(balanced_accuracy_score(y, prediction, **kwargs)) if len(np.unique(y)) == 2 else np.nan,
        "Brier": float(brier_score_loss(y, probability, **kwargs)),
        "Accuracy": float(accuracy_score(y, prediction, **kwargs)),
        "TN": float(tn), "FP": float(fp), "FN": float(fn), "TP": float(tp),
    }


def apply_bh_fdr(probability_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    corrected = np.zeros_like(probability_frame.to_numpy(float))
    rejected = np.zeros_like(corrected, dtype=bool)
    for index, row in enumerate(probability_frame.to_numpy(float)):
        rejected[index], corrected[index], _, _ = multipletests(row, alpha=0.05, method="fdr_bh")
    return rejected.astype(int), corrected


def weighted_calibration(y: np.ndarray, probability: np.ndarray, weights: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.unique(np.quantile(probability, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for index in range(len(edges) - 1):
        mask = bucket == index
        if not mask.any():
            continue
        w = weights[mask]
        rows.append(
            {
                "bin": index + 1, "n": int(mask.sum()), "weight": float(w.sum()),
                "mean_probability": float(np.average(probability[mask], weights=w)),
                "observed_rate": float(np.average(y[mask], weights=w)),
            }
        )
    return pd.DataFrame(rows)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
            )
    return result.replace([np.inf, -np.inf], np.nan)


def sheet_spec(name: str, frame: pd.DataFrame) -> dict[str, object]:
    frame = clean_frame(frame)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append([
            None if pd.isna(value) else (value.item() if isinstance(value, np.generic) else value)
            for value in row
        ])
    return {"name": name, "columns": [str(column) for column in frame.columns], "rows": rows}


def make_figures(predictions: pd.DataFrame, best_model: str) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    colors = {"Normal": "#0072B2", "Abnormal": "#D55E00"}
    panel_letters = "ABCDEF"
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 8.4))
    for row, chromosome in enumerate(CHROMOSOMES):
        label = predictions[f"true_T{chromosome}"].map({0: "Normal", 1: "Abnormal"})
        for column, variable in enumerate((f"Z{chromosome}", f"Z{chromosome}_cal")):
            ax = axes[row, column]
            for group in ("Normal", "Abnormal"):
                values = predictions.loc[label.eq(group), variable]
                sns.kdeplot(values, ax=ax, color=colors[group], lw=1.8, label=f"{group} (n={len(values)})", warn_singular=False)
            ax.axvline(3, color="#555555", ls="--", lw=1)
            ax.set_xlabel(("Raw" if column == 0 else "Calibrated") + f" Z{chromosome}")
            ax.set_ylabel("Density")
            ax.text(-0.14, 1.04, panel_letters[row * 2 + column], transform=ax.transAxes, fontweight="bold")
            ax.legend(frameon=False, fontsize=7)
            sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "raw_vs_calibrated_z.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    weights = patient_weights(predictions)
    for model, color, style in zip(("M1", "M2", "M3"), ("#0072B2", "#D55E00", "#009E73"), ("-", "--", "-."), strict=True):
        precision, recall, _ = precision_recall_curve(
            predictions["true_any"], predictions[f"{model}_prob_any"], sample_weight=weights
        )
        ap = average_precision_score(predictions["true_any"], predictions[f"{model}_prob_any"], sample_weight=weights)
        ax.plot(recall, precision, color=color, ls=style, lw=2, label=f"{DISPLAY_NAME[model]} (AP={ap:.3f})")
    prevalence = float(np.average(predictions["true_any"], weights=weights))
    ax.axhline(prevalence, color="#777777", ls=":", lw=1, label=f"Prevalence={prevalence:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_pr_curves.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(
        predictions["true_any"], predictions[f"{best_model}_pred_any"], labels=[0, 1], sample_weight=weights
    )
    normalized = cm / cm.sum(axis=1, keepdims=True)
    annotations = np.array([[f"{normalized[i, j]:.1%}\n(weight={cm[i, j]:.1f})" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(normalized, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Row proportion"}, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="Observed AB label", xticklabels=["Normal", "Any abnormal"], yticklabels=["Normal", "Any abnormal"])
    ax.set_title(DISPLAY_NAME[best_model])
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "best_model_confusion.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    calibration = weighted_calibration(
        predictions["true_any"].to_numpy(int), predictions[f"{best_model}_prob_any"].to_numpy(float), weights
    )
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 5.2), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot([0, 1], [0, 1], color="#777777", ls="--", lw=1, label="Perfect calibration")
    axes[0].plot(calibration["mean_probability"], calibration["observed_rate"], color="#0072B2", marker="o", lw=2, label=DISPLAY_NAME[best_model])
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed abnormal rate", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].hist(predictions[f"{best_model}_prob_any"], bins=np.linspace(0, 1, 16), weights=weights, color="#56B4E9", edgecolor="white")
    axes[1].set(xlabel="Predicted probability", ylabel="Patient weight")
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "best_model_calibration.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)
    for directory in (DATA_DIR, RAW_DIR, DECISION_DIR, FIGURE_DIR, TMP_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    data, data_audit = prepare_data()
    data = add_raw_absolute(data)
    predictions = data.copy()
    predictions["fold"] = 0
    predictions["qc_low_confidence"] = False
    predictions["qc_reason"] = ""

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_splits = list(splitter.split(data, data["true_any"], groups=data["patient_id"]))
    fold_audit_rows: list[dict[str, object]] = []
    for fold, (_, validation_positions) in enumerate(outer_splits, 1):
        validation = data.iloc[validation_positions]
        predictions.loc[validation.index, "fold"] = fold
        row = {
            "fold": fold, "rows": len(validation), "patients": validation["patient_id"].nunique(),
            "Normal": int(validation["is_normal_reference"].sum()),
            "T13 positive": int(validation["true_T13"].sum()),
            "T18 positive": int(validation["true_T18"].sum()),
            "T21 positive": int(validation["true_T21"].sum()),
            "Any abnormal": int(validation["true_any"].sum()),
        }
        if min(row["T13 positive"], row["T18 positive"], row["T21 positive"]) == 0:
            raise AssertionError("outer fold lacks a chromosome-positive record")
        fold_audit_rows.append(row)
    if predictions["fold"].eq(0).any():
        raise AssertionError("outer fold assignments incomplete")
    if predictions.groupby("patient_id")["fold"].nunique().max() != 1:
        raise AssertionError("patient leakage across outer folds")

    calibration_rows: list[pd.DataFrame] = []
    qc_threshold_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for fold, (train_positions, validation_positions) in enumerate(outer_splits, 1):
        print(f"outer fold {fold}/5", flush=True)
        outer_train = data.iloc[train_positions].copy()
        outer_validation = data.iloc[validation_positions].copy()
        qc_fit, qc_rows = robust_qc_fit(outer_train, fold)
        qc_threshold_rows.extend(qc_rows)
        validation_flags, validation_reasons = robust_qc_apply(outer_validation, qc_fit)
        predictions.loc[outer_validation.index, "qc_low_confidence"] = validation_flags
        predictions.loc[outer_validation.index, "qc_reason"] = validation_reasons

        calibrated_train, calibrated_validation, diagnostics = calibrate_partition(
            outer_train, outer_validation, f"outer_{fold}"
        )
        diagnostics["stage"] = "outer"
        diagnostics["outer_fold"] = fold
        for chromosome in CHROMOSOMES:
            mask = calibrated_validation["is_normal_reference"].eq(1)
            selected = diagnostics["chromosome"].eq(f"T{chromosome}")
            values = calibrated_validation.loc[mask, f"Z{chromosome}_cal"].to_numpy(float)
            diagnostics.loc[selected, "validation_normal_n"] = len(values)
            diagnostics.loc[selected, "validation_normal_cal_median"] = float(np.median(values))
            diagnostics.loc[selected, "validation_normal_cal_mad"] = float(np.median(np.abs(values - np.median(values))))
        calibration_rows.append(diagnostics)
        for chromosome in CHROMOSOMES:
            predictions.loc[outer_validation.index, f"Z{chromosome}_cal"] = calibrated_validation[f"Z{chromosome}_cal"].to_numpy()
            predictions.loc[outer_validation.index, f"abs_Z{chromosome}_cal"] = calibrated_validation[f"abs_Z{chromosome}_cal"].to_numpy()

        inner_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        inner_sets_by_model: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for inner_fold, (inner_train_positions, inner_validation_positions) in enumerate(
            inner_splitter.split(outer_train, outer_train["true_any"], groups=outer_train["patient_id"]), 1
        ):
            inner_train = outer_train.iloc[inner_train_positions].copy()
            inner_validation = outer_train.iloc[inner_validation_positions].copy()
            calibrated_inner_train, calibrated_inner_validation, inner_diagnostics = calibrate_partition(
                inner_train, inner_validation, f"outer_{fold}_inner_{inner_fold}"
            )
            inner_diagnostics["stage"] = "inner"
            inner_diagnostics["outer_fold"] = fold
            inner_diagnostics["inner_fold"] = inner_fold
            calibration_rows.append(inner_diagnostics)
            inner_sets_by_model.append((calibrated_inner_train, calibrated_inner_validation))

        raw_inner_sets = [(data.loc[train.index], data.loc[validation.index]) for train, validation in inner_sets_by_model]
        for model_key in ("M1_raw", "M1", "M2", "M3"):
            inner_sets = raw_inner_sets if model_key == "M1_raw" else inner_sets_by_model
            final_train = outer_train if model_key == "M1_raw" else calibrated_train
            final_validation = outer_validation if model_key == "M1_raw" else calibrated_validation
            features = raw_features() if model_key == "M1_raw" else calibrated_features()
            for chromosome in CHROMOSOMES:
                label = f"true_T{chromosome}"
                parameters, _, threshold, inner_fnr, inner_fpr, inner_risk = tune_label_model(
                    model_key, chromosome, outer_train, inner_sets
                )
                y_train = final_train[label].to_numpy(int)
                estimator = make_estimator(model_key, parameters, y_train)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    estimator.fit(final_train[features], y_train)
                probability = estimator.predict_proba(final_validation[features])[:, 1]
                prediction = (probability >= threshold).astype(int)
                predictions.loc[outer_validation.index, f"{model_key}_prob_T{chromosome}"] = probability
                predictions.loc[outer_validation.index, f"{model_key}_pred_T{chromosome}"] = prediction
                threshold_rows.append(
                    {
                        "fold": fold, "model": DISPLAY_NAME[model_key], "label": f"T{chromosome}",
                        "threshold": threshold, "inner_FNR": inner_fnr, "inner_FPR": inner_fpr,
                        "inner_minimax_risk": inner_risk, "parameters": parameters,
                    }
                )
                selection_rows.append(
                    {
                        "fold": fold, "model": DISPLAY_NAME[model_key], "label": f"T{chromosome}",
                        "selected_parameters": parameters, "feature_set": "raw Z" if model_key == "M1_raw" else "fold-calibrated Z*",
                    }
                )

        raw_probability = pd.DataFrame({f"T{c}": norm.cdf(outer_validation[f"Z{c}"].to_numpy(float)) for c in CHROMOSOMES})
        calibrated_probability = pd.DataFrame({f"T{c}": norm.cdf(calibrated_validation[f"Z{c}_cal"].to_numpy(float)) for c in CHROMOSOMES})
        p_values = pd.DataFrame({f"T{c}": norm.sf(calibrated_validation[f"Z{c}_cal"].to_numpy(float)) for c in CHROMOSOMES})
        fdr_prediction, fdr_q = apply_bh_fdr(p_values)
        for index, chromosome in enumerate(CHROMOSOMES):
            predictions.loc[outer_validation.index, f"B0_prob_T{chromosome}"] = raw_probability.iloc[:, index].to_numpy()
            predictions.loc[outer_validation.index, f"B0_pred_T{chromosome}"] = (outer_validation[f"Z{chromosome}"].to_numpy() > 3).astype(int)
            predictions.loc[outer_validation.index, f"B1_prob_T{chromosome}"] = calibrated_probability.iloc[:, index].to_numpy()
            predictions.loc[outer_validation.index, f"B1_pred_T{chromosome}"] = (calibrated_validation[f"Z{chromosome}_cal"].to_numpy() > 3).astype(int)
            predictions.loc[outer_validation.index, f"B1_FDR_prob_T{chromosome}"] = 1 - fdr_q[:, index]
            predictions.loc[outer_validation.index, f"B1_FDR_pred_T{chromosome}"] = fdr_prediction[:, index]
            threshold_rows.extend(
                [
                    {"fold": fold, "model": DISPLAY_NAME["B0"], "label": f"T{chromosome}", "threshold": 3.0, "inner_FNR": np.nan, "inner_FPR": np.nan, "inner_minimax_risk": np.nan, "parameters": "fixed raw Z rule"},
                    {"fold": fold, "model": DISPLAY_NAME["B1"], "label": f"T{chromosome}", "threshold": 3.0, "inner_FNR": np.nan, "inner_FPR": np.nan, "inner_minimax_risk": np.nan, "parameters": "fixed calibrated Z* rule"},
                    {"fold": fold, "model": DISPLAY_NAME["B1_FDR"], "label": f"T{chromosome}", "threshold": 0.05, "inner_FNR": np.nan, "inner_FPR": np.nan, "inner_minimax_risk": np.nan, "parameters": "BH-FDR within three chromosomes"},
                ]
            )

    predictions["qc_low_confidence"] = predictions["qc_low_confidence"].astype(bool)
    for model_key in MODEL_KEYS:
        chromosome_probabilities = predictions[[f"{model_key}_prob_T{chromosome}" for chromosome in CHROMOSOMES]].to_numpy(float)
        chromosome_predictions = predictions[[f"{model_key}_pred_T{chromosome}" for chromosome in CHROMOSOMES]].to_numpy(int)
        predictions[f"{model_key}_prob_any"] = 1 - np.prod(1 - chromosome_probabilities, axis=1)
        predictions[f"{model_key}_pred_any"] = chromosome_predictions.max(axis=1)

    metric_rows: list[dict[str, object]] = []
    for model_key in MODEL_KEYS:
        for chromosome in CHROMOSOMES:
            label_name = f"T{chromosome}"
            y = predictions[f"true_{label_name}"].to_numpy(int)
            probability = predictions[f"{model_key}_prob_{label_name}"].to_numpy(float)
            prediction = predictions[f"{model_key}_pred_{label_name}"].to_numpy(int)
            for level, weights in (("row-level", None), ("patient-weighted", patient_weights(predictions))):
                metric_rows.append({"model_key": model_key, "Model": DISPLAY_NAME[model_key], "Label": label_name, "Metric level": level, **metric_record(y, probability, prediction, weights)})
        y = predictions["true_any"].to_numpy(int)
        probability = predictions[f"{model_key}_prob_any"].to_numpy(float)
        prediction = predictions[f"{model_key}_pred_any"].to_numpy(int)
        for level, weights in (("row-level", None), ("patient-weighted", patient_weights(predictions))):
            metric_rows.append({"model_key": model_key, "Model": DISPLAY_NAME[model_key], "Label": "Any", "Metric level": level, **metric_record(y, probability, prediction, weights)})
    metrics = pd.DataFrame(metric_rows)
    patient_any = metrics.query("Label == 'Any' and `Metric level` == 'patient-weighted'").copy()

    candidates = patient_any[patient_any["model_key"].isin(["M1", "M2", "M3"])].sort_values(
        ["PR-AUC", "Recall", "F2", "Brier"], ascending=[False, False, False, True]
    )
    top = candidates.iloc[0]
    m1 = candidates[candidates["model_key"].eq("M1")].iloc[0]
    prefer_m1 = (
        float(top["PR-AUC"] - m1["PR-AUC"]) < 0.02
        and abs(float(top["Recall"] - m1["Recall"])) <= 0.03
        and abs(float(top["F2"] - m1["F2"])) <= 0.03
    )
    best_model = "M1" if prefer_m1 else str(top["model_key"])
    candidate_order = [best_model] + [key for key in candidates["model_key"] if key != best_model]
    first_model, second_model, rejected_model = candidate_order

    patient_any["Decision"] = "Benchmark"
    patient_any.loc[patient_any["model_key"].eq("M1_raw"), "Decision"] = "Calibration ablation"
    patient_any.loc[patient_any["model_key"].eq(first_model), "Decision"] = "First recommendation"
    patient_any.loc[patient_any["model_key"].eq(second_model), "Decision"] = "Second recommendation"
    patient_any.loc[patient_any["model_key"].eq(rejected_model), "Decision"] = "Not recommended in round 1"
    decision_comparison = patient_any[[
        "Model", "PR-AUC", "Recall", "Precision", "F2", "Balanced Accuracy", "Brier", "model_key", "Decision"
    ]].rename(columns={"PR-AUC": "Any PR-AUC"})
    chromosome_patient = metrics.query("Label != 'Any' and `Metric level` == 'patient-weighted'")
    for chromosome in CHROMOSOMES:
        lookup = chromosome_patient[(chromosome_patient["Label"].eq(f"T{chromosome}"))].set_index("model_key")["PR-AUC"]
        decision_comparison[f"T{chromosome} PR-AUC"] = decision_comparison["model_key"].map(lookup)
    decision_comparison = decision_comparison.drop(columns="model_key").sort_values("Any PR-AUC", ascending=False)

    calibration_gain = patient_any[patient_any["model_key"].isin(["B0", "B1", "B1_FDR", "M1_raw", "M1"])][
        ["Model", "PR-AUC", "Recall", "Precision", "F2", "Balanced Accuracy", "Brier"]
    ].copy()
    b0 = patient_any.set_index("model_key").loc["B0"]
    b1 = patient_any.set_index("model_key").loc["B1"]
    raw_m1 = patient_any.set_index("model_key").loc["M1_raw"]
    cal_m1 = patient_any.set_index("model_key").loc["M1"]
    calibration_gain["Role"] = [
        "fixed benchmark" if name.startswith("B0") else
        "calibrated fixed benchmark" if "Z*>3" in name else
        "FDR reference" if "FDR" in name else
        "raw-Z ablation" if "M1-raw" in name else "calibrated classifier"
        for name in calibration_gain["Model"]
    ]
    calibration_gain_deltas = pd.DataFrame(
        [
            {
                "Comparison": "B1 calibrated Z*>3 minus B0 raw Z>3",
                **{f"Delta {metric}": float(b1[metric] - b0[metric]) for metric in ["PR-AUC", "Recall", "F2", "Balanced Accuracy", "Brier"]},
            },
            {
                "Comparison": "M1 calibrated minus M1-raw",
                **{f"Delta {metric}": float(cal_m1[metric] - raw_m1[metric]) for metric in ["PR-AUC", "Recall", "F2", "Balanced Accuracy", "Brier"]},
            },
        ]
    )

    repeat_detail = (
        predictions.groupby("technical_repeat_group_id", as_index=False)
        .agg(
            patient_id=("patient_id", "first"), rows=("row_id", "size"),
            distinct_AB_labels=("AB_label", "nunique"),
            distinct_T13=("true_T13", "nunique"), distinct_T18=("true_T18", "nunique"),
            distinct_T21=("true_T21", "nunique"), distinct_any=("true_any", "nunique"),
        )
    )
    repeat_detail = repeat_detail[repeat_detail["rows"].gt(1)].copy()
    repeat_detail["label_inconsistent"] = repeat_detail["distinct_AB_labels"].gt(1)
    inconsistent_rate = float(repeat_detail["label_inconsistent"].mean()) if len(repeat_detail) else np.nan
    inconsistent_row_rate = (
        float(repeat_detail.loc[repeat_detail["label_inconsistent"], "rows"].sum() / repeat_detail["rows"].sum())
        if len(repeat_detail) else np.nan
    )

    subgroup_rows: list[dict[str, object]] = []
    for name, mask in (
        ("normal quality", ~predictions["qc_low_confidence"]),
        ("low quality", predictions["qc_low_confidence"]),
    ):
        subset = predictions.loc[mask]
        if len(subset) and subset["true_any"].nunique() == 2:
            record = metric_record(
                subset["true_any"].to_numpy(int), subset[f"{best_model}_prob_any"].to_numpy(float),
                subset[f"{best_model}_pred_any"].to_numpy(int), patient_weights(subset),
            )
        else:
            record = {key: np.nan for key in ["PR-AUC", "ROC-AUC", "Recall", "Precision", "F1", "F2", "Specificity", "Balanced Accuracy", "Brier", "Accuracy", "TN", "FP", "FN", "TP"]}
        subgroup_rows.append(
            {"subgroup": name, "rows": len(subset), "patients": subset["patient_id"].nunique(), "positive_rows": int(subset["true_any"].sum()), **record}
        )
    subgroup_metrics = pd.DataFrame(subgroup_rows)
    low = subgroup_metrics[subgroup_metrics["subgroup"].eq("low quality")].iloc[0]
    normal_quality = subgroup_metrics[subgroup_metrics["subgroup"].eq("normal quality")].iloc[0]
    if int(low["rows"]) < 10 or int(low["positive_rows"]) < 3:
        quality_conclusion = "低质量样本量或阳性数过少，无法可靠判断是否明显恶化"
    else:
        degraded = (normal_quality["PR-AUC"] - low["PR-AUC"] > 0.05) or (normal_quality["Recall"] - low["Recall"] > 0.10)
        quality_conclusion = "是（描述性指标下降）" if degraded else "否（未见明显描述性下降）"

    all_calibration = pd.concat(calibration_rows, ignore_index=True)
    threshold_table = pd.DataFrame(threshold_rows)
    selection_table = pd.DataFrame(selection_rows)
    fold_audit = pd.DataFrame(fold_audit_rows)
    fold_assignments = predictions[["row_id", "patient_id", "fold", "true_T13", "true_T18", "true_T21", "true_any"]]
    prediction_columns = [
        "row_id", "patient_id", "fold", "technical_repeat_group_id", "true_T13", "true_T18", "true_T21", "true_any",
        "qc_low_confidence", "qc_reason", "Z13", "Z18", "Z21", "Z13_cal", "Z18_cal", "Z21_cal",
    ]
    for model_key in MODEL_KEYS:
        prediction_columns.extend([f"{model_key}_prob_T{c}" for c in CHROMOSOMES])
        prediction_columns.extend([f"{model_key}_pred_T{c}" for c in CHROMOSOMES])
        prediction_columns.extend([f"{model_key}_prob_any", f"{model_key}_pred_any"])
    prediction_export = predictions[prediction_columns].sort_values("row_id")

    any_metrics = metrics[metrics["Label"].eq("Any")].drop(columns="model_key")
    chromosome_metrics = metrics[metrics["Label"].ne("Any")].drop(columns="model_key")
    best_any_patient = patient_any.set_index("model_key").loc[best_model]
    confusion_any_patient = pd.DataFrame(
        [[best_any_patient["TN"], best_any_patient["FP"]], [best_any_patient["FN"], best_any_patient["TP"]]],
        index=["Observed Normal", "Observed Any abnormal"], columns=["Predicted Normal", "Predicted Any abnormal"],
    ).reset_index(names="Observed")
    best_any_row = metrics.query("Label == 'Any' and `Metric level` == 'row-level'").set_index("model_key").loc[best_model]
    confusion_any_row = pd.DataFrame(
        [[best_any_row["TN"], best_any_row["FP"]], [best_any_row["FN"], best_any_row["TP"]]],
        index=["Observed Normal", "Observed Any abnormal"], columns=["Predicted Normal", "Predicted Any abnormal"],
    ).reset_index(names="Observed")
    best_chromosome_confusion = chromosome_patient[chromosome_patient["model_key"].eq(best_model)][
        ["Label", "TN", "FP", "FN", "TP", "Recall", "Precision", "F2", "Balanced Accuracy"]
    ]

    candidate_chromosome = chromosome_patient[chromosome_patient["model_key"].isin(["M1", "M2", "M3"])]
    chromosome_best = {}
    for chromosome in CHROMOSOMES:
        row = candidate_chromosome[candidate_chromosome["Label"].eq(f"T{chromosome}")].sort_values("PR-AUC", ascending=False).iloc[0]
        chromosome_best[chromosome] = (str(row["Model"]), float(row["PR-AUC"]))

    m1_pr_delta = float(cal_m1["PR-AUC"] - raw_m1["PR-AUC"])
    calibration_conclusion = (
        "作为候选模块值得保留，但尚未证明具有一致判别增益：固定规则的 Recall/F2 提高，"
        f"校准 M1 的 Recall/F2 与 Balanced Accuracy 提高，但 PR-AUC 变化为 {m1_pr_delta:+.4f}"
    )
    summary = f"""# Q4 第一轮模型筛选摘要

## 校准是否有效

原始 Z>3 的 Any Recall / F2 = {b0['Recall']:.4f} / {b0['F2']:.4f}

校准 Z*>3 的 Any Recall / F2 = {b1['Recall']:.4f} / {b1['F2']:.4f}

M1-raw PR-AUC = {raw_m1['PR-AUC']:.4f}

M1-calibrated PR-AUC = {cal_m1['PR-AUC']:.4f}

结论：Z 重新校准是否值得保留？{calibration_conclusion}。以上均为严格孕妇级外层 OOF、patient-weighted 指标。

## 模型排名

| Rank | Model | Any PR-AUC | Recall | F2 | Balanced Accuracy | Brier |
|---:|---|---:|---:|---:|---:|---:|
"""
    ranked = patient_any[patient_any["model_key"].isin(["M1", "M2", "M3"])].copy()
    rank_order = [first_model, second_model, rejected_model]
    for rank, key in enumerate(rank_order, 1):
        row = ranked[ranked["model_key"].eq(key)].iloc[0]
        summary += f"| {rank} | {row['Model']} | {row['PR-AUC']:.4f} | {row['Recall']:.4f} | {row['F2']:.4f} | {row['Balanced Accuracy']:.4f} | {row['Brier']:.4f} |\n"
    summary += f"""

## 各染色体

T13 最佳模型及 PR-AUC = {chromosome_best[13][0]} / {chromosome_best[13][1]:.4f}

T18 最佳模型及 PR-AUC = {chromosome_best[18][0]} / {chromosome_best[18][1]:.4f}

T21 最佳模型及 PR-AUC = {chromosome_best[21][0]} / {chromosome_best[21][1]:.4f}

## 标签噪声

低质量样本数量 = {int(predictions['qc_low_confidence'].sum())} 条记录（{predictions.loc[predictions['qc_low_confidence'], 'patient_id'].nunique()} 位孕妇）

技术重复标签不一致率 = {inconsistent_rate:.4%}（按具有重复记录的 technical-repeat group 计；按重复记录加权为 {inconsistent_row_rate:.4%}）

低质量样本上的模型表现是否明显恶化 = {quality_conclusion}

## 最终建议

第一推荐：{DISPLAY_NAME[first_model]}

第二推荐：{DISPLAY_NAME[second_model]}

不推荐：{DISPLAY_NAME[rejected_model]}（本轮相对前两者无决策优势）

建议继续把“质量门控 → 条件 Z 校准 → 多标签概率分类 → 风险阈值”作为 Q4 候选统一框架，但下一轮必须保留 M1-raw 对照，不能把条件 Z 校准视为已证实的必要环节。本轮仅评估对题目 AB 判定标签的复现能力；AB 不是经核型验证的真值。Q4 尚未冻结，未建立三级判定。
"""
    (DECISION_DIR / "q4_summary.md").write_text(summary, encoding="utf-8")

    versions = pd.DataFrame(
        [
            {"component": "random_seed", "version_or_value": SEED},
            {"component": "outer_cv", "version_or_value": "StratifiedGroupKFold(5), groups=patient_id"},
            {"component": "inner_cv", "version_or_value": "StratifiedGroupKFold(3), groups=patient_id"},
            {"component": "pandas", "version_or_value": importlib.metadata.version("pandas")},
            {"component": "numpy", "version_or_value": np.__version__},
            {"component": "scipy", "version_or_value": scipy.__version__},
            {"component": "scikit-learn", "version_or_value": sklearn.__version__},
            {"component": "lightgbm", "version_or_value": lightgbm.__version__},
            {"component": "mgcv family", "version_or_value": ", ".join(sorted(all_calibration["family"].unique()))},
        ]
    )
    data_audit_table = pd.DataFrame(
        [{"item": key, "value": value} for key, value in data_audit.items()]
        + [
            {"item": "technical_repeat_groups", "value": len(repeat_detail)},
            {"item": "technical_repeat_inconsistent_rate", "value": inconsistent_rate},
            {"item": "qc_low_confidence_rows", "value": int(predictions["qc_low_confidence"].sum())},
            {"item": "best_model", "value": DISPLAY_NAME[best_model]},
        ]
    )
    payload = {
        "workbooks": [
            {"path": "raw/calibration_diagnostics.xlsx", "sheets": [sheet_spec("Z calibration", all_calibration), sheet_spec("QC thresholds", pd.DataFrame(qc_threshold_rows))]},
            {"path": "raw/fold_assignments.xlsx", "sheets": [sheet_spec("Assignments", fold_assignments), sheet_spec("Fold audit", fold_audit)]},
            {"path": "raw/fold_predictions.xlsx", "sheets": [sheet_spec("Predictions", prediction_export)]},
            {"path": "raw/chromosome_metrics_raw.xlsx", "sheets": [sheet_spec("Metrics", chromosome_metrics)]},
            {"path": "raw/any_abnormal_metrics_raw.xlsx", "sheets": [sheet_spec("Metrics", any_metrics)]},
            {"path": "raw/thresholds_by_fold.xlsx", "sheets": [sheet_spec("Thresholds", threshold_table)]},
            {"path": "raw/q4_all_raw.xlsx", "sheets": [sheet_spec("Data audit", data_audit_table), sheet_spec("Technical repeats", repeat_detail), sheet_spec("QC subgroups", subgroup_metrics), sheet_spec("Model selection", selection_table), sheet_spec("Versions", versions)]},
            {"path": "decision/q4_model_comparison.xlsx", "sheets": [sheet_spec("Model comparison", decision_comparison)]},
            {"path": "decision/q4_calibration_gain.xlsx", "sheets": [sheet_spec("Calibration gain", calibration_gain), sheet_spec("Paired deltas", calibration_gain_deltas)]},
            {"path": "decision/q4_best_model_confusion.xlsx", "sheets": [sheet_spec("Any patient weighted", confusion_any_patient), sheet_spec("Any row level", confusion_any_row), sheet_spec("Chromosomes", best_chromosome_confusion)]},
        ]
    }
    PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    predictions.to_csv(DATA_DIR / "q4_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(DATA_DIR / "q4_metrics.csv", index=False, encoding="utf-8-sig")
    make_figures(predictions, best_model)
    print(f"best_model={DISPLAY_NAME[best_model]}")
    print(f"payload={PAYLOAD_PATH}")


if __name__ == "__main__":
    main()
