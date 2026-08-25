from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from run_q4_models import (
    CHROMOSOMES,
    DATA_DIR,
    OUTPUT_DIR,
    Q4_ROOT,
    SEED,
    add_raw_absolute,
    calibrate_partition,
    metric_record,
    patient_weights,
    sheet_spec,
)
from run_q4_round2 import (
    calibration_statistics,
    estimator_for,
    feature_columns,
    grid_for,
    multiclass_metrics,
    tune_label,
)


FIRST_OOF = DATA_DIR / "q4_oof_predictions.csv"
ROUND2_OOF = DATA_DIR / "q4_round2_oof_predictions.csv"
MODEL_DATA = DATA_DIR / "q4_model_data.csv"
DECISION_DIR = OUTPUT_DIR / "decision_layer"
RAW_OUTPUT = DATA_DIR / "q4_decision_layer_oof_predictions.csv"
PAYLOAD_PATH = OUTPUT_DIR / ".q4_decision_layer_payload.json"
BINARY_RECALL_TARGETS = (0.85, 0.90, 0.95)
REJECT_LEVELS = (0.05, 0.10, 0.15)


def candidate_thresholds(probability: np.ndarray) -> np.ndarray:
    unique = np.unique(np.clip(probability, 0.0, 1.0))
    midpoints = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else np.array([])
    return np.unique(np.r_[0.0, 1.0, unique, midpoints])


def tune_direct_any(
    training: pd.DataFrame,
    inner_sets: list[tuple[pd.DataFrame, pd.DataFrame]],
    weights: np.ndarray,
) -> dict[str, object]:
    features = feature_columns("M1_U")
    best_score = -np.inf
    best_parameters: dict[str, object] | None = None
    best_oof: np.ndarray | None = None
    for parameters in grid_for("M1_U"):
        oof = np.full(len(training), np.nan)
        for inner_train, inner_validation in inner_sets:
            y_train = inner_train["true_any"].to_numpy(int)
            estimator = estimator_for("M1_U", parameters, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator.fit(inner_train[features], y_train)
            positions = training.index.get_indexer(inner_validation.index)
            oof[positions] = estimator.predict_proba(inner_validation[features])[:, 1]
        if np.isnan(oof).any():
            raise AssertionError("direct Any inner OOF probabilities are incomplete")
        score = float(average_precision_score(training["true_any"], oof, sample_weight=weights))
        if score > best_score + 1e-12:
            best_score, best_parameters, best_oof = score, parameters, oof
    assert best_parameters is not None and best_oof is not None
    return {"parameters": best_parameters, "oof_probability": best_oof, "inner_pr_auc": best_score}


def predict_direct_any(
    training: pd.DataFrame, validation: pd.DataFrame, parameters: dict[str, object]
) -> np.ndarray:
    features = feature_columns("M1_U")
    estimator = estimator_for("M1_U", parameters, training["true_any"].to_numpy(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(training[features], training["true_any"])
    return estimator.predict_proba(validation[features])[:, 1]


def constrained_binary_threshold(
    y: np.ndarray, probability: np.ndarray, weights: np.ndarray, recall_target: float
) -> tuple[float, float, float]:
    best: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds(probability):
        prediction = probability >= threshold
        recall = float(weights[(y == 1) & prediction].sum() / weights[y == 1].sum())
        if recall + 1e-12 < recall_target:
            continue
        specificity = float(weights[(y == 0) & (~prediction)].sum() / weights[y == 0].sum())
        candidate = (specificity, float(threshold), recall)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[0]


def reject_thresholds(
    y: np.ndarray, probability: np.ndarray, weights: np.ndarray, alpha: float, beta: float
) -> tuple[float, float, float, float, float]:
    thresholds = candidate_thresholds(probability)
    positive_weight = weights[y == 1].sum()
    negative_weight = weights[y == 0].sum()
    lower = []
    upper = []
    for threshold in thresholds:
        fnr = float(weights[(y == 1) & (probability < threshold)].sum() / positive_weight)
        fpr = float(weights[(y == 0) & (probability >= threshold)].sum() / negative_weight)
        if fnr <= alpha + 1e-12:
            lower.append((float(threshold), fnr))
        if fpr <= beta + 1e-12:
            upper.append((float(threshold), fpr))
    best: tuple[tuple[float, float, float, float], float, float, float, float, float] | None = None
    for tau_l, fnr in lower:
        for tau_h, fpr in upper:
            if tau_l >= tau_h:
                continue
            definite = (probability < tau_l) | (probability >= tau_h)
            coverage = float(weights[definite].sum() / weights.sum())
            key = (coverage, -(tau_h - tau_l), tau_l, -tau_h)
            if best is None or key > best[0]:
                best = (key, tau_l, tau_h, fnr, fpr, coverage)
    assert best is not None
    return best[1], best[2], best[3], best[4], best[5]


def localization_threshold(
    y: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float]:
    best: tuple[float, float, float, float, float] | None = None
    for threshold in candidate_thresholds(probability):
        prediction = probability >= threshold
        f1 = float(f1_score(y, prediction, sample_weight=weights, zero_division=0))
        recall = float(recall_score(y, prediction, sample_weight=weights, zero_division=0))
        precision = float(precision_score(y, prediction, sample_weight=weights, zero_division=0))
        candidate = (f1, recall, precision, -abs(float(threshold) - 0.5), float(threshold))
        if best is None or candidate[:4] > best[:4]:
            best = candidate
    assert best is not None
    return best[4], best[0], best[1], best[2]


def apply_three_level(
    probability: np.ndarray, tau_l: float, tau_h: float, qc_low: np.ndarray
) -> np.ndarray:
    prediction = np.where(
        probability < tau_l,
        "Normal",
        np.where(probability >= tau_h, "Abnormal", "Suspicious"),
    ).astype(object)
    prediction[qc_low & (prediction != "Abnormal")] = "Suspicious"
    return prediction.astype(str)


def selective_metrics(
    frame: pd.DataFrame, prediction_column: str, weights: np.ndarray
) -> dict[str, float]:
    decision = frame[prediction_column].to_numpy(str)
    definite = decision != "Suspicious"
    definite_prediction = (decision[definite] == "Abnormal").astype(int)
    definite_truth = frame["true_any"].to_numpy(int)[definite]
    definite_weights = weights[definite]
    accuracy = float(accuracy_score(definite_truth, definite_prediction, sample_weight=definite_weights))
    balanced = (
        float(balanced_accuracy_score(definite_truth, definite_prediction, sample_weight=definite_weights))
        if len(np.unique(definite_truth)) == 2
        else np.nan
    )
    normal = decision == "Normal"
    abnormal = decision == "Abnormal"
    truth = frame["true_any"].to_numpy(int)
    npv = float(weights[normal & (truth == 0)].sum() / weights[normal].sum()) if normal.any() else np.nan
    ppv = float(weights[abnormal & (truth == 1)].sum() / weights[abnormal].sum()) if abnormal.any() else np.nan
    total = weights.sum()
    return {
        "Normal proportion": float(weights[normal].sum() / total),
        "Suspicious rate": float(weights[decision == "Suspicious"].sum() / total),
        "Abnormal proportion": float(weights[abnormal].sum() / total),
        "Coverage": float(weights[definite].sum() / total),
        "Selective Accuracy": accuracy,
        "Selective Balanced Accuracy": balanced,
        "Definite-normal NPV": npv,
        "Definite-abnormal PPV": ppv,
    }


def weighted_group_rate(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[mask].sum() / weights.sum()) if weights.sum() else np.nan


def subtype_from_bits(bits: tuple[int, int, int]) -> str:
    mapping = {
        (1, 0, 0): "T13",
        (0, 1, 0): "T18",
        (0, 0, 1): "T21",
        (1, 1, 0): "T13T18",
        (1, 0, 1): "T13T21",
        (0, 1, 1): "T18T21",
        (1, 1, 1): "T13T18T21",
    }
    return mapping.get(bits, "Abnormal-subtype-uncertain")


def final_class(row: pd.Series) -> str:
    if row["three_level_prediction"] == "Normal":
        return "Normal"
    if row["three_level_prediction"] == "Suspicious":
        return "Suspicious"
    return subtype_from_bits(tuple(int(row[f"pred_T{chromosome}"]) for chromosome in CHROMOSOMES))


def make_figures(frame: pd.DataFrame, binary_comparison: pd.DataFrame) -> None:
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    weights = patient_weights(frame)
    y = frame["true_any"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for probability, label, color, style in (
        (frame["pi_any"].to_numpy(float), "Direct Any M1-U", "#0072B2", "-"),
        (frame["M1_U_prob_any"].to_numpy(float), "Old chromosome OR", "#D55E00", "--"),
    ):
        precision, recall, _ = precision_recall_curve(y, probability, sample_weight=weights)
        ap = average_precision_score(y, probability, sample_weight=weights)
        ax.plot(recall, precision, color=color, linestyle=style, linewidth=2, label=f"{label} (AP={ap:.3f})")
    ax.axhline(np.average(y, weights=weights), color="#666666", linestyle=":", label="Weighted prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(DECISION_DIR / "any_abnormal_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    bins = np.linspace(0, 1, 26)
    for value, label, color in ((0, "True normal", "#56B4E9"), (1, "True abnormal", "#D55E00")):
        mask = y == value
        group_weights = weights[mask] / weights[mask].sum()
        ax.hist(frame.loc[mask, "pi_any"], bins=bins, weights=group_weights, alpha=0.55, color=color, label=label)
    tau_l = float(frame["tau_L"].median())
    tau_h = float(frame["tau_H"].median())
    ax.axvspan(tau_l, tau_h, color="#E69F00", alpha=0.12, label="Suspicious interval (median thresholds)")
    ax.axvline(tau_l, color="#009E73", linestyle="--", linewidth=1.8, label=f"median tau_L={tau_l:.3f}")
    ax.axvline(tau_h, color="#CC79A7", linestyle="-.", linewidth=1.8, label=f"median tau_H={tau_h:.3f}")
    ax.set(xlabel="Direct Any-abnormal probability", ylabel="Within-class weighted proportion", xlim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(DECISION_DIR / "three_level_probability_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True)
    for ax, (label, prediction) in zip(
        axes,
        (
            ("Old chromosome MM to OR", frame["M1_U_pred_any_MM"].to_numpy(int)),
            ("New direct Any gate", frame["binary_prediction"].to_numpy(int)),
        ),
        strict=True,
    ):
        matrix = confusion_matrix(y, prediction, labels=[0, 1], sample_weight=weights, normalize="true")
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"],
            ax=ax,
        )
        ba = float(binary_comparison.loc[binary_comparison["Rule"].eq(label), "Balanced Accuracy"].iloc[0])
        ax.set_title(f"{label}\nBalanced accuracy={ba:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(DECISION_DIR / "old_vs_new_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)
    base = add_raw_absolute(pd.read_csv(MODEL_DATA))
    first = pd.read_csv(FIRST_OOF)
    round2 = pd.read_csv(ROUND2_OOF)
    if not (base["row_id"].is_unique and first["row_id"].is_unique and round2["row_id"].is_unique):
        raise AssertionError("row_id must be unique")
    if not (set(base["row_id"]) == set(first["row_id"]) == set(round2["row_id"])):
        raise AssertionError("Q4 source row_id sets differ")

    first_by_id = first.set_index("row_id")
    round2_by_id = round2.set_index("row_id")
    base["fold"] = base["row_id"].map(first_by_id["fold"]).astype(int)
    base["qc_low_confidence"] = base["row_id"].map(first_by_id["qc_low_confidence"]).astype(bool)
    base["true_AB"] = base["AB_label"].fillna("Normal").replace("", "Normal")
    for column in ["patient_id", "true_T13", "true_T18", "true_T21", "true_any"]:
        expected = base["row_id"].map(round2_by_id[column]).to_numpy()
        if not np.array_equal(base[column].to_numpy(), expected):
            raise AssertionError(f"round2 mismatch: {column}")
    if base.groupby("patient_id")["fold"].nunique().max() != 1:
        raise AssertionError("patient outer folds are not grouped")

    result = base.copy()
    for chromosome in CHROMOSOMES:
        result[f"p{chromosome}"] = result["row_id"].map(round2_by_id[f"M1_U_prob_T{chromosome}"])
        result[f"old_pred_T{chromosome}"] = result["row_id"].map(round2_by_id[f"M1_U_pred_T{chromosome}_MM"]).astype(int)
    result["M1_U_prob_any"] = result["row_id"].map(round2_by_id["M1_U_prob_any"])
    result["M1_U_pred_any_MM"] = result["row_id"].map(round2_by_id["M1_U_pred_any_MM"]).astype(int)

    parameter_rows: list[dict[str, object]] = []
    binary_threshold_rows: list[dict[str, object]] = []
    reject_threshold_rows: list[dict[str, object]] = []
    localization_threshold_rows: list[dict[str, object]] = []

    for fold in range(1, 6):
        print(f"decision outer fold {fold}/5", flush=True)
        train = base[base["fold"].ne(fold)].copy()
        validation = base[base["fold"].eq(fold)].copy()
        calibrated_train, calibrated_validation, _ = calibrate_partition(train, validation, f"outer_{fold}")
        inner_sets: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        for inner_fold, (train_pos, validation_pos) in enumerate(
            splitter.split(train, train["true_any"], groups=train["patient_id"]), 1
        ):
            inner_train = train.iloc[train_pos].copy()
            inner_validation = train.iloc[validation_pos].copy()
            cal_train, cal_validation, _ = calibrate_partition(
                inner_train, inner_validation, f"outer_{fold}_inner_{inner_fold}"
            )
            inner_sets.append((cal_train, cal_validation))

        weights = patient_weights(train)
        direct = tune_direct_any(train, inner_sets, weights)
        outer_probability = predict_direct_any(calibrated_train, calibrated_validation, direct["parameters"])
        result.loc[validation.index, "pi_any"] = outer_probability
        parameter_rows.append(
            {
                "fold": fold,
                "Model": "Direct Any M1-U",
                "Selected parameters": direct["parameters"],
                "Inner PR-AUC": direct["inner_pr_auc"],
            }
        )

        y_train = train["true_any"].to_numpy(int)
        inner_probability = np.asarray(direct["oof_probability"], dtype=float)
        for recall_target in BINARY_RECALL_TARGETS:
            threshold, recall, specificity = constrained_binary_threshold(
                y_train, inner_probability, weights, recall_target
            )
            column = f"binary_prediction_r{int(recall_target * 100)}"
            result.loc[validation.index, column] = (outer_probability >= threshold).astype(int)
            binary_threshold_rows.append(
                {
                    "fold": fold,
                    "Recall target": recall_target,
                    "Threshold": threshold,
                    "Training Recall": recall,
                    "Training Specificity": specificity,
                }
            )
            if np.isclose(recall_target, 0.90):
                result.loc[validation.index, "binary_threshold"] = threshold

        qc_validation = validation["qc_low_confidence"].to_numpy(bool)
        main_tau_h: float | None = None
        for level in REJECT_LEVELS:
            tau_l, tau_h, fnr, fpr, coverage = reject_thresholds(
                y_train, inner_probability, weights, level, level
            )
            column = f"three_level_a{int(level * 100):02d}"
            result.loc[validation.index, column] = apply_three_level(
                outer_probability, tau_l, tau_h, qc_validation
            )
            reject_threshold_rows.append(
                {
                    "fold": fold,
                    "alpha=beta": level,
                    "tau_L": tau_l,
                    "tau_H": tau_h,
                    "Training FNR at tau_L": fnr,
                    "Training FPR at tau_H": fpr,
                    "Training Coverage": coverage,
                }
            )
            if np.isclose(level, 0.10):
                result.loc[validation.index, "tau_L"] = tau_l
                result.loc[validation.index, "tau_H"] = tau_h
                main_tau_h = tau_h

        assert main_tau_h is not None
        for chromosome in CHROMOSOMES:
            tuned = tune_label("M1_U", chromosome, train, inner_sets, weights)
            screened = inner_probability >= main_tau_h
            threshold, f1, recall, precision = localization_threshold(
                train.loc[screened, f"true_T{chromosome}"].to_numpy(int),
                np.asarray(tuned["oof_probability"])[screened],
                weights[screened],
            )
            result.loc[validation.index, f"loc_threshold_T{chromosome}"] = threshold
            localization_threshold_rows.append(
                {
                    "fold": fold,
                    "Chromosome": f"T{chromosome}",
                    "Threshold": threshold,
                    "Training F1": f1,
                    "Training Recall": recall,
                    "Training Precision": precision,
                    "Selected parameters": tuned["parameters"],
                    "Selection population": "direct-gate Abnormal outer-training OOF",
                }
            )

    required = ["pi_any", "binary_threshold", "tau_L", "tau_H"]
    if result[required].isna().any().any():
        raise AssertionError("outer OOF decision values are incomplete")
    result["binary_prediction"] = result["binary_prediction_r90"].astype(int)
    result["three_level_prediction"] = result["three_level_a10"].astype(str)
    for chromosome in CHROMOSOMES:
        result[f"pred_T{chromosome}"] = (
            result["three_level_prediction"].eq("Abnormal")
            & result[f"p{chromosome}"].ge(result[f"loc_threshold_T{chromosome}"])
        ).astype(int)
    suspected = result[[f"p{chromosome}" for chromosome in CHROMOSOMES]].idxmax(axis=1).str.replace("p", "T", regex=False)
    result["top_suspected_chromosome"] = np.where(
        result["three_level_prediction"].eq("Suspicious"), suspected, None
    )
    result["final_AB_prediction"] = result.apply(final_class, axis=1)

    weights = patient_weights(result)
    y = result["true_any"].to_numpy(int)
    direct_probability = result["pi_any"].to_numpy(float)
    direct_prediction = result["binary_prediction"].to_numpy(int)
    direct_metrics = {
        "Model": "Direct Any M1-U",
        **metric_record(y, direct_probability, direct_prediction, weights),
        **calibration_statistics(y, direct_probability, weights),
    }
    probability_table = pd.DataFrame([direct_metrics])[
        ["Model", "PR-AUC", "ROC-AUC", "Brier", "Calibration intercept", "Calibration slope", "ECE"]
    ]

    binary_rows = []
    for recall_target in BINARY_RECALL_TARGETS:
        prediction = result[f"binary_prediction_r{int(recall_target * 100)}"].to_numpy(int)
        binary_rows.append(
            {
                "Rule": f"Direct Any Recall>={recall_target:.2f}",
                "Recall target": recall_target,
                **metric_record(y, direct_probability, prediction, weights),
            }
        )
    binary_sensitivity = pd.DataFrame(binary_rows)

    binary_comparison = pd.DataFrame(
        [
            {
                "Rule": "Old chromosome MM to OR",
                **metric_record(
                    y,
                    result["M1_U_prob_any"].to_numpy(float),
                    result["M1_U_pred_any_MM"].to_numpy(int),
                    weights,
                ),
            },
            {
                "Rule": "New direct Any gate",
                **metric_record(y, direct_probability, direct_prediction, weights),
            },
        ]
    )

    selective_rows = []
    for level in REJECT_LEVELS:
        column = f"three_level_a{int(level * 100):02d}"
        selective_rows.append({"alpha=beta": level, **selective_metrics(result, column, weights)})
    selective_table = pd.DataFrame(selective_rows)
    main_selective = selective_table.loc[np.isclose(selective_table["alpha=beta"], 0.10)].iloc[0]

    qc_rows = []
    for name, mask in (
        ("QC normal", ~result["qc_low_confidence"].to_numpy(bool)),
        ("QC low confidence", result["qc_low_confidence"].to_numpy(bool)),
    ):
        subgroup_weights = weights[mask]
        decision = result.loc[mask, "three_level_prediction"].to_numpy(str)
        qc_rows.append(
            {
                "QC group": name,
                "N rows": int(mask.sum()),
                "N patients": int(result.loc[mask, "patient_id"].nunique()),
                "Coverage": weighted_group_rate(decision != "Suspicious", subgroup_weights),
                "Suspicious rate": weighted_group_rate(decision == "Suspicious", subgroup_weights),
                "Normal rate": weighted_group_rate(decision == "Normal", subgroup_weights),
                "Abnormal rate": weighted_group_rate(decision == "Abnormal", subgroup_weights),
            }
        )
    qc_table = pd.DataFrame(qc_rows)

    localization_rows = []
    multilabel_rows = []
    abnormal = y == 1
    abnormal_weights = weights[abnormal]
    true_bits = result.loc[abnormal, [f"true_T{chromosome}" for chromosome in CHROMOSOMES]].to_numpy(int)
    for rule, prefix in (("Old chromosome MM", "old_pred_"), ("New two-stage localization", "pred_")):
        predicted_bits = result.loc[abnormal, [f"{prefix}T{chromosome}" for chromosome in CHROMOSOMES]].to_numpy(int)
        for index, chromosome in enumerate(CHROMOSOMES):
            localization_rows.append(
                {
                    "Rule": rule,
                    "Chromosome": f"T{chromosome}",
                    "Precision": float(precision_score(true_bits[:, index], predicted_bits[:, index], sample_weight=abnormal_weights, zero_division=0)),
                    "Recall": float(recall_score(true_bits[:, index], predicted_bits[:, index], sample_weight=abnormal_weights, zero_division=0)),
                    "F1": float(f1_score(true_bits[:, index], predicted_bits[:, index], sample_weight=abnormal_weights, zero_division=0)),
                    "Interpretation": "Exploratory; very few T21 observations" if chromosome == 21 else "Primary localization metric",
                }
            )
        exact = np.all(true_bits == predicted_bits, axis=1)
        multilabel_rows.append(
            {
                "Rule": rule,
                "Population": "true Any-abnormal records",
                "Hamming loss": float(hamming_loss(true_bits, predicted_bits, sample_weight=abnormal_weights)),
                "Exact-match accuracy": float(np.average(exact, weights=abnormal_weights)),
                "N rows": int(abnormal.sum()),
                "N patients": int(result.loc[abnormal, "patient_id"].nunique()),
            }
        )
    localization_table = pd.DataFrame(localization_rows)
    multilabel_table = pd.DataFrame(multilabel_rows)

    conservative = multiclass_metrics(result["true_AB"], result["final_AB_prediction"])
    undecided = result["final_AB_prediction"].isin(["Suspicious", "Abnormal-subtype-uncertain"])
    decided = ~undecided
    decided_metrics = multiclass_metrics(
        result.loc[decided, "true_AB"], result.loc[decided, "final_AB_prediction"]
    )
    national_table = pd.DataFrame(
        [
            {
                "Protocol": "Conservative exact classification",
                **conservative,
                "Coverage": 1.0,
                "N evaluated": len(result),
                "Status": "Suspicious and subtype-uncertain are independent predicted classes",
            },
            {
                "Protocol": "Decided-only",
                **decided_metrics,
                "Coverage": float(decided.mean()),
                "N evaluated": int(decided.sum()),
                "Status": "Reports coverage; does not hide undecided records",
            },
            {
                "Protocol": "National-prize paper RF (external reference)",
                "Accuracy": 0.9125,
                "Weighted Recall": 0.9125,
                "Weighted F1": 0.8706,
                "Weighted Precision": np.nan,
                "Coverage": np.nan,
                "N evaluated": np.nan,
                "Status": "Paper-reported 70/30 row-level split; not directly comparable",
            },
        ]
    )

    confusion_rows = []
    for rule, prediction in (
        ("Old chromosome MM to OR", result["M1_U_pred_any_MM"].to_numpy(int)),
        ("New direct Any gate", direct_prediction),
    ):
        raw = confusion_matrix(y, prediction, labels=[0, 1])
        weighted = confusion_matrix(y, prediction, labels=[0, 1], sample_weight=weights)
        for true_index, true_label in enumerate(("Normal", "Abnormal")):
            for pred_index, pred_label in enumerate(("Normal", "Abnormal")):
                confusion_rows.append(
                    {
                        "Rule": rule,
                        "True": true_label,
                        "Predicted": pred_label,
                        "Row count": int(raw[true_index, pred_index]),
                        "Patient-weighted count": float(weighted[true_index, pred_index]),
                    }
                )
    confusion_table = pd.DataFrame(confusion_rows)

    raw_columns = [
        "row_id", "patient_id", "true_any", "true_AB", "pi_any", "p13", "p18", "p21",
        "binary_threshold", "binary_prediction", "tau_L", "tau_H", "three_level_prediction",
        "pred_T13", "pred_T18", "pred_T21", "final_AB_prediction", "top_suspected_chromosome",
        "qc_low_confidence", "fold",
    ]
    result[raw_columns].to_csv(RAW_OUTPUT, index=False, encoding="utf-8-sig")

    main_localization = localization_table[localization_table["Rule"].eq("New two-stage localization")].set_index("Chromosome")
    main_multilabel = multilabel_table[multilabel_table["Rule"].eq("New two-stage localization")].iloc[0]
    old = binary_comparison.set_index("Rule").loc["Old chromosome MM to OR"]
    new = binary_comparison.set_index("Rule").loc["New direct Any gate"]
    direct_supported = bool(new["Recall"] >= 0.85 and new["Specificity"] > old["Specificity"] and new["Precision"] > old["Precision"])
    three_supported = bool(main_selective["Selective Accuracy"] > new["Accuracy"] and main_selective["Coverage"] > 0.50)

    summary = f"""# Q4 最终决策层摘要

## Any-abnormal probability

PR-AUC = {direct_metrics['PR-AUC']:.6f}

Brier = {direct_metrics['Brier']:.6f}

Calibration slope = {direct_metrics['Calibration slope']:.6f}

Calibration intercept = {direct_metrics['Calibration intercept']:.6f}

## 二分类决策

旧 OR：Recall={old['Recall']:.6f}, Specificity={old['Specificity']:.6f}, Precision={old['Precision']:.6f}, F2={old['F2']:.6f}, Balanced Accuracy={old['Balanced Accuracy']:.6f}

新 direct Any：Recall={new['Recall']:.6f}, Specificity={new['Specificity']:.6f}, Precision={new['Precision']:.6f}, F2={new['F2']:.6f}, Balanced Accuracy={new['Balanced Accuracy']:.6f}

## 三级判定（alpha=beta=0.10）

tau_L（outer-fold median） = {result['tau_L'].median():.6f}

tau_H（outer-fold median） = {result['tau_H'].median():.6f}

Normal比例 = {main_selective['Normal proportion']:.6f}

Suspicious比例 = {main_selective['Suspicious rate']:.6f}

Abnormal比例 = {main_selective['Abnormal proportion']:.6f}

Coverage = {main_selective['Coverage']:.6f}

Selective Accuracy = {main_selective['Selective Accuracy']:.6f}

NPV = {main_selective['Definite-normal NPV']:.6f}

PPV = {main_selective['Definite-abnormal PPV']:.6f}

严格 OOF 使用每个 outer fold 单独由 outer-training OOF 选择的阈值；上面 tau_L/tau_H 仅报告五折中位数。

## 染色体定位

T13 F1 = {main_localization.loc['T13', 'F1']:.6f}

T18 F1 = {main_localization.loc['T18', 'F1']:.6f}

T21 F1 = {main_localization.loc['T21', 'F1']:.6f}（探索性；样本极少）

Exact-match accuracy = {main_multilabel['Exact-match accuracy']:.6f}

## 最终判断

是否冻结 M1-U = 是

是否采用 direct Any-abnormal gate = {'是' if direct_supported else '否'}

是否采用 Normal/Suspicious/Abnormal 三级决策 = {'是' if three_supported else '否'}

是否保留 chromosome-specific 第二阶段定位 = 是（T21 仅探索性报告）

Q4 是否可以正式冻结 = {'是' if direct_supported and three_supported else '否'}
"""
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    (DECISION_DIR / "q4_decision_summary.md").write_text(summary, encoding="utf-8")

    payload = {
        "workbooks": [
            {
                "path": "decision_layer/any_model_metrics.xlsx",
                "sheets": [sheet_spec("OOF probability metrics", probability_table), sheet_spec("Fold parameters", pd.DataFrame(parameter_rows))],
            },
            {
                "path": "decision_layer/binary_threshold_comparison.xlsx",
                "sheets": [sheet_spec("Old vs direct", binary_comparison), sheet_spec("Recall sensitivity", binary_sensitivity), sheet_spec("Thresholds by fold", pd.DataFrame(binary_threshold_rows))],
            },
            {
                "path": "decision_layer/three_level_decision.xlsx",
                "sheets": [sheet_spec("Selective metrics", selective_table), sheet_spec("QC subgroups", qc_table), sheet_spec("Thresholds by fold", pd.DataFrame(reject_threshold_rows)), sheet_spec("OOF raw", result[raw_columns])],
            },
            {
                "path": "decision_layer/chromosome_localization.xlsx",
                "sheets": [sheet_spec("Chromosome metrics", localization_table), sheet_spec("Multilabel summary", multilabel_table), sheet_spec("Thresholds by fold", pd.DataFrame(localization_threshold_rows))],
            },
            {
                "path": "decision_layer/old_vs_new_decision.xlsx",
                "sheets": [sheet_spec("Binary comparison", binary_comparison), sheet_spec("Confusion detail", confusion_table)],
            },
            {
                "path": "decision_layer/q4_final_national_prize_comparable_metrics.xlsx",
                "sheets": [sheet_spec("Comparable metrics", national_table)],
            },
        ]
    }
    PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    make_figures(result, binary_comparison)
    print(f"direct_any_pr_auc={direct_metrics['PR-AUC']:.6f}")
    print(f"direct_any_brier={direct_metrics['Brier']:.6f}")
    print(f"payload={PAYLOAD_PATH}")


if __name__ == "__main__":
    main()
