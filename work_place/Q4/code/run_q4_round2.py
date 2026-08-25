from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_q4_models import (
    CHROMOSOMES,
    DATA_DIR,
    DECISION_DIR,
    FIGURE_DIR,
    OUTPUT_DIR,
    Q4_ROOT,
    SEED,
    add_raw_absolute,
    calibrate_partition,
    calibrated_features,
    make_estimator,
    metric_record,
    minimax_threshold,
    parameter_grid,
    patient_weights,
    raw_features,
    sheet_spec,
    weighted_calibration,
)


BASELINE_PREDICTIONS = DATA_DIR / "q4_oof_predictions.csv"
MODEL_DATA = DATA_DIR / "q4_model_data.csv"
ROUND2_PREDICTIONS = DATA_DIR / "q4_round2_oof_predictions.csv"
PAPER_PREDICTIONS = DATA_DIR / "q4_round2_paper_protocol_predictions.csv"
PAYLOAD_PATH = OUTPUT_DIR / ".q4_round2_workbook_payload.json"
SUMMARY_PATH = DECISION_DIR / "q4_round2_summary.md"

DISPLAY = {
    "M1_raw": "M1-raw",
    "M1": "M1-cal",
    "M1_U": "M1-U",
    "M1_H": "M1-H",
    "M1_HP": "M1-HP",
    "M2": "Random Forest",
    "M3": "LightGBM",
    "RF_paper": "RF-paper-style",
}
STRICT_KEYS = ("M1_raw", "M1", "M1_U", "M1_H", "M1_HP", "M2", "M3", "RF_paper")
NEW_KEYS = ("M1_U", "M1_H", "M1_HP", "RF_paper")
BASELINE_KEYS = ("M1_raw", "M1", "M2", "M3")
ELASTIC_KEYS = {"M1_raw", "M1", "M1_U", "M1_H", "M1_HP"}
PAPER_REFERENCE = {
    "Model": "国一论文 Random Forest（论文报告）",
    "Accuracy": 0.9125,
    "Weighted Recall": 0.9125,
    "Weighted F1": 0.8706,
    "Weighted Precision": np.nan,
    "Evaluation protocol": "70/30 row-level split reported in paper",
    "Reference status": "External paper value; not ranked",
}


def hybrid_features() -> list[str]:
    return [
        "Z13", "Z18", "Z21", "Z13_cal", "Z18_cal", "Z21_cal",
        "abs_Z13", "abs_Z18", "abs_Z21", "abs_Z13_cal", "abs_Z18_cal", "abs_Z21_cal",
        *calibrated_features()[6:],
    ]


def feature_columns(model_key: str) -> list[str]:
    if model_key == "M1_raw":
        return raw_features()
    if model_key in {"M1_H", "M1_HP"}:
        return hybrid_features()
    return calibrated_features()


def logistic_estimator(parameters: dict[str, object], balanced: bool) -> Pipeline:
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        class_weight="balanced" if balanced else None,
        max_iter=5000,
        random_state=SEED,
        n_jobs=1,
        **parameters,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def rf_paper_estimator() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )


def estimator_for(model_key: str, parameters: dict[str, object], y: np.ndarray) -> Pipeline:
    if model_key in {"M1_raw", "M1", "M1_U", "M1_H", "M1_HP"}:
        return logistic_estimator(parameters, balanced=model_key in {"M1_raw", "M1", "M1_HP"})
    if model_key in {"M2", "M3"}:
        return make_estimator(model_key, parameters, y)
    if model_key == "RF_paper":
        return rf_paper_estimator()
    raise KeyError(model_key)


def grid_for(model_key: str) -> list[dict[str, object]]:
    if model_key in ELASTIC_KEYS:
        return parameter_grid("M1")
    if model_key in {"M2", "M3"}:
        return parameter_grid(model_key)
    if model_key == "RF_paper":
        return [{}]
    raise KeyError(model_key)


def fit_platt(scores: np.ndarray, y: np.ndarray, weights: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(
        penalty=None, solver="lbfgs", class_weight=None, random_state=SEED, max_iter=2000
    )
    calibrator.fit(scores.reshape(-1, 1), y, sample_weight=weights)
    return calibrator


def f2_threshold(
    y: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float]:
    unique = np.unique(np.clip(probability, 0, 1))
    thresholds = np.unique(np.r_[0.0, 1.0, unique, (unique[:-1] + unique[1:]) / 2])
    best: tuple[float, float, float, float, float] | None = None
    for threshold in thresholds:
        prediction = probability >= threshold
        f2 = float(fbeta_score(y, prediction, beta=2, zero_division=0, sample_weight=weights))
        recall = float(recall_score(y, prediction, zero_division=0, sample_weight=weights))
        precision = float(precision_score(y, prediction, zero_division=0, sample_weight=weights))
        candidate = (-f2, -precision, -recall, abs(float(threshold) - 0.5), float(threshold))
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    assert best is not None
    return best[4], -best[0], -best[2], -best[1]


def tune_label(
    model_key: str,
    chromosome: int,
    training: pd.DataFrame,
    inner_sets: list[tuple[pd.DataFrame, pd.DataFrame]],
    weights: np.ndarray,
) -> dict[str, object]:
    label = f"true_T{chromosome}"
    features = feature_columns(model_key)
    best_ap = -math.inf
    best_parameters: dict[str, object] | None = None
    best_oof: np.ndarray | None = None
    best_scores: np.ndarray | None = None
    for parameters in grid_for(model_key):
        oof_probability = np.full(len(training), np.nan)
        oof_score = np.full(len(training), np.nan)
        for inner_train, inner_validation in inner_sets:
            y_train = inner_train[label].to_numpy(int)
            if len(np.unique(y_train)) != 2:
                raise ValueError(f"inner training split lacks both classes: {model_key} {label}")
            estimator = estimator_for(model_key, parameters, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator.fit(inner_train[features], y_train)
            positions = training.index.get_indexer(inner_validation.index)
            oof_probability[positions] = estimator.predict_proba(inner_validation[features])[:, 1]
            if model_key == "M1_HP":
                oof_score[positions] = estimator.decision_function(inner_validation[features])
        if np.isnan(oof_probability).any():
            raise AssertionError("inner OOF probabilities are incomplete")
        ranking = oof_score if model_key == "M1_HP" else oof_probability
        score = float(average_precision_score(training[label], ranking, sample_weight=weights))
        if score > best_ap + 1e-12:
            best_ap = score
            best_parameters = parameters
            best_oof = oof_probability
            best_scores = oof_score if model_key == "M1_HP" else None
    assert best_parameters is not None and best_oof is not None
    platt = None
    if model_key == "M1_HP":
        assert best_scores is not None and not np.isnan(best_scores).any()
        platt = fit_platt(best_scores, training[label].to_numpy(int), weights)
        best_oof = platt.predict_proba(best_scores.reshape(-1, 1))[:, 1]
    mm_threshold, mm_fnr, mm_fpr, mm_risk = minimax_threshold(
        training[label].to_numpy(int), best_oof, weights
    )
    f2_value, f2_score, f2_recall, f2_precision = f2_threshold(
        training[label].to_numpy(int), best_oof, weights
    )
    return {
        "parameters": best_parameters,
        "oof_probability": best_oof,
        "platt": platt,
        "inner_pr_auc": best_ap,
        "minimax_threshold": mm_threshold,
        "minimax_fnr": mm_fnr,
        "minimax_fpr": mm_fpr,
        "minimax_risk": mm_risk,
        "f2_threshold": f2_value,
        "f2_score": f2_score,
        "f2_recall": f2_recall,
        "f2_precision": f2_precision,
    }


def outer_predict(
    model_key: str,
    chromosome: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tuned: dict[str, object],
) -> np.ndarray:
    label = f"true_T{chromosome}"
    features = feature_columns(model_key)
    estimator = estimator_for(model_key, tuned["parameters"], train[label].to_numpy(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(train[features], train[label])
    if model_key == "M1_HP":
        scores = estimator.decision_function(validation[features])
        return tuned["platt"].predict_proba(scores.reshape(-1, 1))[:, 1]
    return estimator.predict_proba(validation[features])[:, 1]


def choose_threshold_strategy(frame: pd.DataFrame, model_key: str) -> tuple[str, dict[str, dict[str, float]]]:
    y = frame["true_any"].to_numpy(int)
    probability = frame[f"{model_key}_prob_any"].to_numpy(float)
    weights = patient_weights(frame)
    metrics = {}
    for strategy in ("Minimax", "F2-optimal"):
        prediction = frame[f"{model_key}_pred_any_{'MM' if strategy == 'Minimax' else 'F2'}"].to_numpy(int)
        metrics[strategy] = metric_record(y, probability, prediction, weights)
    mm, f2 = metrics["Minimax"], metrics["F2-optimal"]
    choose_f2 = (
        f2["Recall"] >= mm["Recall"] - 0.03
        and f2["F2"] >= mm["F2"]
        and (f2["Precision"] >= mm["Precision"] + 0.02 or f2["Balanced Accuracy"] >= mm["Balanced Accuracy"] + 0.02)
    )
    return ("F2-optimal" if choose_f2 else "Minimax"), metrics


def calibration_statistics(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=SEED)
    model.fit(logit(clipped).reshape(-1, 1), y, sample_weight=weights)
    bins = np.minimum((clipped * 10).astype(int), 9)
    ece = 0.0
    total_weight = weights.sum()
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += weights[mask].sum() / total_weight * abs(
                np.average(y[mask], weights=weights[mask])
                - np.average(clipped[mask], weights=weights[mask])
            )
    return {
        "Calibration intercept": float(model.intercept_[0]),
        "Calibration slope": float(model.coef_[0, 0]),
        "ECE": float(ece),
    }


def bits_to_class(frame: pd.DataFrame, prefix: str) -> pd.Series:
    mapping = {
        (0, 0, 0): "Normal",
        (1, 0, 0): "T13",
        (0, 1, 0): "T18",
        (0, 0, 1): "T21",
        (1, 1, 0): "T13T18",
        (1, 0, 1): "T13T21",
        (0, 1, 1): "T18T21",
        (1, 1, 1): "T13T18T21",
    }
    tuples = list(zip(*(frame[f"{prefix}T{chromosome}"].astype(int) for chromosome in CHROMOSOMES), strict=True))
    return pd.Series([mapping[value] for value in tuples], index=frame.index)


def multiclass_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    accuracy = float(accuracy_score(y_true, y_pred))
    weighted_recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    if not np.isclose(accuracy, weighted_recall, atol=1e-12):
        raise AssertionError("weighted recall must equal accuracy in single-label multiclass evaluation")
    return {
        "Accuracy": accuracy,
        "Weighted Recall": weighted_recall,
        "Weighted F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Weighted Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def add_any_columns(frame: pd.DataFrame, model_key: str) -> None:
    probabilities = frame[[f"{model_key}_prob_T{chromosome}" for chromosome in CHROMOSOMES]].to_numpy(float)
    frame[f"{model_key}_prob_any"] = 1 - np.prod(1 - probabilities, axis=1)
    for suffix in ("MM", "F2"):
        predictions = frame[[f"{model_key}_pred_T{chromosome}_{suffix}" for chromosome in CHROMOSOMES]].to_numpy(int)
        frame[f"{model_key}_pred_any_{suffix}"] = predictions.max(axis=1)


def strict_round2(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = data.copy()
    threshold_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for fold in range(1, 6):
        print(f"strict outer fold {fold}/5", flush=True)
        train = data[data["fold"].ne(fold)].copy()
        validation = data[data["fold"].eq(fold)].copy()
        calibrated_train, calibrated_validation, _ = calibrate_partition(train, validation, f"outer_{fold}")
        inner_sets: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        # The saved outer fold is grouped; inner tuning must also remain grouped.
        grouped_inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        for inner_fold, (train_pos, validation_pos) in enumerate(
            grouped_inner.split(train, train["true_any"], groups=train["patient_id"]), 1
        ):
            inner_train = train.iloc[train_pos].copy()
            inner_validation = train.iloc[validation_pos].copy()
            cal_train, cal_validation, _ = calibrate_partition(
                inner_train, inner_validation, f"outer_{fold}_inner_{inner_fold}"
            )
            inner_sets.append((cal_train, cal_validation))
        weights = patient_weights(train)
        for model_key in NEW_KEYS:
            for chromosome in CHROMOSOMES:
                tuned = tune_label(model_key, chromosome, train, inner_sets, weights)
                probability = outer_predict(
                    model_key, chromosome, calibrated_train, calibrated_validation, tuned
                )
                result.loc[validation.index, f"{model_key}_prob_T{chromosome}"] = probability
                result.loc[validation.index, f"{model_key}_pred_T{chromosome}_MM"] = (
                    probability >= tuned["minimax_threshold"]
                ).astype(int)
                result.loc[validation.index, f"{model_key}_pred_T{chromosome}_F2"] = (
                    probability >= tuned["f2_threshold"]
                ).astype(int)
                threshold_rows.extend(
                    [
                        {
                            "fold": fold,
                            "Model": DISPLAY[model_key],
                            "Label": f"T{chromosome}",
                            "Strategy": "Minimax",
                            "Threshold": tuned["minimax_threshold"],
                            "Training FNR": tuned["minimax_fnr"],
                            "Training FPR": tuned["minimax_fpr"],
                            "Training minimax risk": tuned["minimax_risk"],
                            "Training F2": np.nan,
                            "Training Recall": np.nan,
                            "Training Precision": np.nan,
                        },
                        {
                            "fold": fold,
                            "Model": DISPLAY[model_key],
                            "Label": f"T{chromosome}",
                            "Strategy": "F2-optimal",
                            "Threshold": tuned["f2_threshold"],
                            "Training FNR": np.nan,
                            "Training FPR": np.nan,
                            "Training minimax risk": np.nan,
                            "Training F2": tuned["f2_score"],
                            "Training Recall": tuned["f2_recall"],
                            "Training Precision": tuned["f2_precision"],
                        },
                    ]
                )
                parameter_rows.append(
                    {
                        "fold": fold,
                        "Model": DISPLAY[model_key],
                        "Label": f"T{chromosome}",
                        "Selected parameters": tuned["parameters"],
                        "Inner PR-AUC": tuned["inner_pr_auc"],
                        "Platt intercept": float(tuned["platt"].intercept_[0]) if tuned["platt"] is not None else np.nan,
                        "Platt slope": float(tuned["platt"].coef_[0, 0]) if tuned["platt"] is not None else np.nan,
                    }
                )
    for model_key in NEW_KEYS:
        add_any_columns(result, model_key)
    return result, pd.DataFrame(threshold_rows), pd.DataFrame(parameter_rows)


def fit_row_protocol_model(
    model_key: str,
    training: pd.DataFrame,
    test: pd.DataFrame,
    calibrated_training: pd.DataFrame,
    calibrated_test: pd.DataFrame,
    inner_raw: list[tuple[pd.DataFrame, pd.DataFrame]],
    inner_calibrated: list[tuple[pd.DataFrame, pd.DataFrame]],
    strategy: str,
) -> pd.DataFrame:
    output = test[["row_id", "patient_id", "true_T13", "true_T18", "true_T21", "true_any"]].copy()
    weights = np.ones(len(training), dtype=float)
    tune_training = training if model_key == "M1_raw" else calibrated_training
    tune_sets = inner_raw if model_key == "M1_raw" else inner_calibrated
    final_training = training if model_key == "M1_raw" else calibrated_training
    final_test = test if model_key == "M1_raw" else calibrated_test
    for chromosome in CHROMOSOMES:
        tuned = tune_label(model_key, chromosome, tune_training, tune_sets, weights)
        probability = outer_predict(model_key, chromosome, final_training, final_test, tuned)
        threshold = tuned["f2_threshold"] if strategy == "F2-optimal" else tuned["minimax_threshold"]
        output[f"{model_key}_prob_T{chromosome}"] = probability
        output[f"{model_key}_pred_T{chromosome}"] = (probability >= threshold).astype(int)
    probabilities = output[[f"{model_key}_prob_T{chromosome}" for chromosome in CHROMOSOMES]].to_numpy(float)
    predictions = output[[f"{model_key}_pred_T{chromosome}" for chromosome in CHROMOSOMES]].to_numpy(int)
    output[f"{model_key}_prob_any"] = 1 - np.prod(1 - probabilities, axis=1)
    output[f"{model_key}_pred_any"] = predictions.max(axis=1)
    return output


def paper_protocol(
    data: pd.DataFrame, final_candidate: str, strict_strategy: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    true_class = bits_to_class(data, "true_")
    train_index, test_index = train_test_split(
        np.arange(len(data)), test_size=0.30, random_state=SEED, stratify=true_class
    )
    training, test = data.iloc[train_index].copy(), data.iloc[test_index].copy()
    calibrated_training, calibrated_test, _ = calibrate_partition(
        training, test, "round2_paper_outer"
    )
    inner_raw: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    inner_calibrated: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (train_pos, validation_pos) in enumerate(splitter.split(training, training["true_any"]), 1):
        inner_train = training.iloc[train_pos].copy()
        inner_validation = training.iloc[validation_pos].copy()
        cal_train, cal_validation, _ = calibrate_partition(
            inner_train, inner_validation, f"round2_paper_inner_{fold}"
        )
        inner_raw.append((inner_train, inner_validation))
        inner_calibrated.append((cal_train, cal_validation))
    outputs = []
    rows = []
    for model_key in (final_candidate, "M2", "M3", "RF_paper"):
        prediction = fit_row_protocol_model(
            model_key,
            training,
            test,
            calibrated_training,
            calibrated_test,
            inner_raw,
            inner_calibrated,
            strict_strategy[model_key],
        )
        outputs.append(prediction.set_index("row_id"))
        y_true = bits_to_class(prediction, "true_")
        y_pred = bits_to_class(prediction, f"{model_key}_pred_")
        row = {
            "Model": DISPLAY[model_key],
            **multiclass_metrics(y_true, y_pred),
            "Evaluation protocol": "70/30 row-level split",
            "Reference status": "Reference-only; possible same-patient leakage",
        }
        rows.append(row)
    combined = outputs[0]
    for output in outputs[1:]:
        columns = [column for column in output.columns if column not in combined.columns]
        combined = combined.join(output[columns], how="inner")
    combined = combined.reset_index()
    overlap = set(training["patient_id"]) & set(test["patient_id"])
    audit = {
        "train_rows": len(training),
        "test_rows": len(test),
        "train_patients": training["patient_id"].nunique(),
        "test_patients": test["patient_id"].nunique(),
        "patients_in_both_train_and_test": len(overlap),
        "same_patient_leakage_possible": bool(overlap),
        "random_state": SEED,
        "stratified_by": "observed AB multiclass label",
    }
    return pd.DataFrame(rows), combined, audit


def make_figures(
    strict: pd.DataFrame,
    paper_predictions: pd.DataFrame,
    final_candidate: str,
    national_strict: pd.DataFrame,
    national_paper: pd.DataFrame,
) -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    colors = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    weights = patient_weights(strict)
    for model_key, color in zip(("M1_raw", "M1", "M1_U", "M1_H", "M1_HP"), colors, strict=True):
        calibration = weighted_calibration(
            strict["true_any"].to_numpy(int), strict[f"{model_key}_prob_any"].to_numpy(float), weights
        )
        brier = metric_record(
            strict["true_any"].to_numpy(int), strict[f"{model_key}_prob_any"].to_numpy(float),
            strict[f"{model_key}_pred_any"].to_numpy(int), weights
        )["Brier"]
        ax.plot(
            calibration["mean_probability"], calibration["observed_rate"], marker="o", lw=1.8,
            color=color, label=f"{DISPLAY[model_key]} (Brier={brier:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1, label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed abnormal rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "round2_probability_calibration.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    models = [final_candidate, "M2", "M3", "RF_paper"]
    plot_rows = []
    for model_key in models:
        strict_row = national_strict[national_strict["Model"].eq(DISPLAY[model_key])].iloc[0]
        paper_row = national_paper[national_paper["Model"].eq(DISPLAY[model_key])].iloc[0]
        for protocol, row in (("Patient-level Group-CV", strict_row), ("70/30 row-level", paper_row)):
            for metric in ("Accuracy", "Weighted F1"):
                plot_rows.append({"Model": DISPLAY[model_key], "Protocol": protocol, "Metric": metric, "Value": row[metric]})
    plot = pd.DataFrame(plot_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, metric in zip(axes, ("Accuracy", "Weighted F1"), strict=True):
        selected = plot[plot["Metric"].eq(metric)]
        sns.barplot(
            data=selected, x="Model", y="Value", hue="Protocol", palette=["#0072B2", "#D55E00"], ax=ax
        )
        ax.set(title=metric, xlabel="", ylabel="Score" if metric == "Accuracy" else "", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=25)
        if metric == "Weighted F1":
            ax.legend_.remove()
        else:
            ax.legend(frameon=False, fontsize=7)
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "strict_vs_paper_protocol_metrics.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)
    baseline = pd.read_csv(BASELINE_PREDICTIONS)
    data = add_raw_absolute(pd.read_csv(MODEL_DATA))
    fold_map = baseline.set_index("row_id")["fold"]
    data["fold"] = data["row_id"].map(fold_map).astype(int)
    if data["fold"].isna().any() or data.groupby("patient_id")["fold"].nunique().max() != 1:
        raise AssertionError("saved first-round outer folds are invalid")
    for column in ["patient_id", "true_T13", "true_T18", "true_T21", "true_any"]:
        expected = baseline.set_index("row_id").loc[data["row_id"], column].to_numpy()
        if not np.array_equal(data[column].to_numpy(), expected):
            raise AssertionError(f"first-round data mismatch: {column}")

    strict_new, threshold_table, parameter_table = strict_round2(data)
    strict = baseline.copy().set_index("row_id")
    new_columns = [
        column for column in strict_new.columns
        if any(column.startswith(f"{key}_") for key in NEW_KEYS)
    ]
    strict = strict.join(strict_new.set_index("row_id")[new_columns], how="left").reset_index()

    strategies = {key: "Minimax" for key in BASELINE_KEYS}
    strategy_metrics: list[dict[str, object]] = []
    for model_key in NEW_KEYS:
        strategy, detail = choose_threshold_strategy(strict, model_key)
        strategies[model_key] = strategy
        suffix = "F2" if strategy == "F2-optimal" else "MM"
        for chromosome in CHROMOSOMES:
            strict[f"{model_key}_pred_T{chromosome}"] = strict[f"{model_key}_pred_T{chromosome}_{suffix}"].astype(int)
        strict[f"{model_key}_pred_any"] = strict[f"{model_key}_pred_any_{suffix}"].astype(int)
        for name, metrics in detail.items():
            strategy_metrics.append({"Model": DISPLAY[model_key], "Strategy": name, **metrics})

    weights = patient_weights(strict)
    main_rows = []
    for model_key in STRICT_KEYS:
        probability = strict[f"{model_key}_prob_any"].to_numpy(float)
        prediction = strict[f"{model_key}_pred_any"].to_numpy(int)
        metrics = metric_record(strict["true_any"].to_numpy(int), probability, prediction, weights)
        main_rows.append(
            {
                "model_key": model_key,
                "Model": DISPLAY[model_key],
                **metrics,
                **calibration_statistics(strict["true_any"].to_numpy(int), probability, weights),
                "Threshold strategy": strategies[model_key],
            }
        )
    main_table = pd.DataFrame(main_rows)

    raw_row = main_table.set_index("model_key").loc["M1_raw"]
    qualified = main_table[
        main_table["model_key"].isin(["M1_U", "M1_H", "M1_HP"])
        & main_table["PR-AUC"].ge(raw_row["PR-AUC"] - 0.02)
        & main_table["Brier"].le(raw_row["Brier"] - 0.02)
    ].sort_values(["PR-AUC", "Brier"], ascending=[False, True])
    if len(qualified):
        first_candidate = str(qualified.iloc[0]["model_key"])
        second_candidate = str(qualified.iloc[1]["model_key"]) if len(qualified) > 1 else "M1_raw"
    else:
        first_candidate = "M1_raw"
        hybrids = main_table[main_table["model_key"].isin(["M1_H", "M1_HP", "M1_U"])].sort_values(
            ["PR-AUC", "Brier"], ascending=[False, True]
        )
        second_candidate = str(hybrids.iloc[0]["model_key"])
    main_table["Decision"] = "Comparison"
    main_table.loc[main_table["model_key"].eq(first_candidate), "Decision"] = "First recommendation"
    main_table.loc[main_table["model_key"].eq(second_candidate), "Decision"] = "Second recommendation"
    main_table.loc[main_table["model_key"].eq("RF_paper"), "Decision"] = "Paper-style reference; not tuned"
    main_table = main_table[
        [
            "Model", "PR-AUC", "ROC-AUC", "Recall", "Precision", "F1", "F2", "Balanced Accuracy",
            "Specificity", "Brier", "Calibration intercept", "Calibration slope", "ECE",
            "Threshold strategy", "Decision", "model_key",
        ]
    ].sort_values("PR-AUC", ascending=False)

    true_multiclass = bits_to_class(strict, "true_")
    strict_rows = []
    for model_key in STRICT_KEYS:
        predicted = bits_to_class(strict, f"{model_key}_pred_")
        strict_rows.append(
            {
                "Model": DISPLAY[model_key],
                **multiclass_metrics(true_multiclass, predicted),
                "Evaluation protocol": "Patient-level Group-CV",
                "Reference status": "Strict OOF; used for our evaluation",
            }
        )
    national_strict = pd.concat([pd.DataFrame(strict_rows), pd.DataFrame([PAPER_REFERENCE])], ignore_index=True)

    national_paper, paper_predictions, paper_audit = paper_protocol(data, first_candidate, strategies)
    national_csv = pd.concat(
        [
            national_strict.assign(Sheet="strict_group_cv"),
            national_paper.assign(Sheet="paper_protocol_reference"),
        ],
        ignore_index=True,
    )
    national_csv.to_csv(
        DECISION_DIR / "q4_national_prize_comparable_metrics.csv", index=False, encoding="utf-8-sig"
    )

    strict_lookup = national_strict.set_index("Model")
    paper_lookup = national_paper.set_index("Model")
    main_lookup = main_table.set_index("Model")
    final_name = DISPLAY[first_candidate]
    compact = pd.DataFrame(
        [
            {
                "方法": final_name,
                "验证方式": "Patient-level Group-CV",
                "Accuracy": strict_lookup.loc[final_name, "Accuracy"],
                "Recall": strict_lookup.loc[final_name, "Weighted Recall"],
                "F1": strict_lookup.loc[final_name, "Weighted F1"],
                "PR-AUC": main_lookup.loc[final_name, "PR-AUC"],
                "Brier": main_lookup.loc[final_name, "Brier"],
            },
            {
                "方法": DISPLAY["RF_paper"],
                "验证方式": "Patient-level Group-CV",
                "Accuracy": strict_lookup.loc[DISPLAY["RF_paper"], "Accuracy"],
                "Recall": strict_lookup.loc[DISPLAY["RF_paper"], "Weighted Recall"],
                "F1": strict_lookup.loc[DISPLAY["RF_paper"], "Weighted F1"],
                "PR-AUC": main_lookup.loc[DISPLAY["RF_paper"], "PR-AUC"],
                "Brier": main_lookup.loc[DISPLAY["RF_paper"], "Brier"],
            },
            {
                "方法": "国一论文 Random Forest",
                "验证方式": "论文报告 70/30 row-level split",
                "Accuracy": 0.9125,
                "Recall": 0.9125,
                "F1": 0.8706,
                "PR-AUC": "—",
                "Brier": "—",
            },
            {
                "方法": final_name,
                "验证方式": "70/30 row-level split（仅参考）",
                "Accuracy": paper_lookup.loc[final_name, "Accuracy"],
                "Recall": paper_lookup.loc[final_name, "Weighted Recall"],
                "F1": paper_lookup.loc[final_name, "Weighted F1"],
                "PR-AUC": average_precision_score(
                    paper_predictions["true_any"], paper_predictions[f"{first_candidate}_prob_any"]
                ),
                "Brier": metric_record(
                    paper_predictions["true_any"].to_numpy(int),
                    paper_predictions[f"{first_candidate}_prob_any"].to_numpy(float),
                    paper_predictions[f"{first_candidate}_pred_any"].to_numpy(int),
                    None,
                )["Brier"],
            },
        ]
    )
    compact_notes = pd.DataFrame(
        {
            "说明": [
                "国奖论文中的 Recall/F1 为多分类加权口径。",
                "PR-AUC/Brier 未由该论文报告，因此对应位置记为“—”。",
                "不同验证协议的数值不可直接视为同等泛化能力。",
                "70/30 row-level 结果仅用于协议差异解释，可能存在同一孕妇泄漏。",
            ]
        }
    )

    round2_export = strict[
        [
            "row_id", "patient_id", "fold", "true_T13", "true_T18", "true_T21", "true_any",
            *[
                column
                for model_key in NEW_KEYS
                for column in [
                    *[f"{model_key}_prob_T{chromosome}" for chromosome in CHROMOSOMES],
                    *[f"{model_key}_pred_T{chromosome}_MM" for chromosome in CHROMOSOMES],
                    *[f"{model_key}_pred_T{chromosome}_F2" for chromosome in CHROMOSOMES],
                    f"{model_key}_prob_any", f"{model_key}_pred_any_MM", f"{model_key}_pred_any_F2",
                    *[f"{model_key}_pred_T{chromosome}" for chromosome in CHROMOSOMES],
                    f"{model_key}_pred_any",
                ]
            ],
        ]
    ]
    round2_export.to_csv(ROUND2_PREDICTIONS, index=False, encoding="utf-8-sig")
    paper_predictions.to_csv(PAPER_PREDICTIONS, index=False, encoding="utf-8-sig")

    protocol_delta = {
        "Accuracy": float(paper_lookup.loc[final_name, "Accuracy"] - strict_lookup.loc[final_name, "Accuracy"]),
        "Weighted F1": float(paper_lookup.loc[final_name, "Weighted F1"] - strict_lookup.loc[final_name, "Weighted F1"]),
    }
    main_metric_lookup = main_table.set_index("model_key")
    raw_metrics = main_metric_lookup.loc["M1_raw"]
    cal_metrics = main_metric_lookup.loc["M1"]
    u_metrics = main_metric_lookup.loc["M1_U"]
    h_metrics = main_metric_lookup.loc["M1_H"]
    hp_metrics = main_metric_lookup.loc["M1_HP"]
    summary = f"""# Q4 第二轮定向改进摘要

## 概率改进

M1-raw PR-AUC / Brier = {raw_metrics['PR-AUC']:.4f} / {raw_metrics['Brier']:.4f}

M1-cal PR-AUC / Brier = {cal_metrics['PR-AUC']:.4f} / {cal_metrics['Brier']:.4f}

M1-U PR-AUC / Brier = {u_metrics['PR-AUC']:.4f} / {u_metrics['Brier']:.4f}

M1-H PR-AUC / Brier = {h_metrics['PR-AUC']:.4f} / {h_metrics['Brier']:.4f}

M1-HP PR-AUC / Brier = {hp_metrics['PR-AUC']:.4f} / {hp_metrics['Brier']:.4f}

## 验证协议影响

最终候选 = {final_name}

Accuracy 变化（70/30 减 strict）= {protocol_delta['Accuracy']:+.4f}

Weighted F1 变化（70/30 减 strict）= {protocol_delta['Weighted F1']:+.4f}

row-level train/test 共同孕妇数 = {paper_audit['patients_in_both_train_and_test']}

## 最终建议

第一推荐：{DISPLAY[first_candidate]}

第二推荐：{DISPLAY[second_candidate]}

是否保留 Z 校准层：{'是，M1-U 使用 calibrated Z*，M1-H 作为 hybrid 次选' if first_candidate == 'M1_U' else '是，作为 hybrid 输入保留'}

下一步是否已经可以冻结 Q4：可以；建议经用户确认后冻结第一推荐与第二推荐，不由本脚本自动执行冻结。
"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    payload = {
        "workbooks": [
            {
                "path": "decision/q4_round2_model_comparison.xlsx",
                "sheets": [
                    sheet_spec("Model comparison", main_table.drop(columns="model_key")),
                    sheet_spec("Probability calibration", main_table[["Model", "Brier", "Calibration intercept", "Calibration slope", "ECE"]]),
                    sheet_spec("Threshold comparison", pd.DataFrame(strategy_metrics)),
                    sheet_spec("Thresholds by fold", threshold_table),
                    sheet_spec("Parameters by fold", parameter_table),
                    sheet_spec("Paper split audit", pd.DataFrame([paper_audit])),
                ],
            },
            {
                "path": "decision/q4_national_prize_comparable_metrics.xlsx",
                "sheets": [
                    sheet_spec("strict_group_cv", national_strict),
                    sheet_spec("paper_protocol_reference", national_paper),
                ],
            },
            {
                "path": "decision/q4_compact_comparison.xlsx",
                "sheets": [sheet_spec("Compact comparison", compact), sheet_spec("Notes", compact_notes)],
            },
        ]
    }
    PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    make_figures(strict, paper_predictions, first_candidate, national_strict, national_paper)
    print(f"first_candidate={DISPLAY[first_candidate]}")
    print(f"second_candidate={DISPLAY[second_candidate]}")
    print(f"payload={PAYLOAD_PATH}")


if __name__ == "__main__":
    main()
