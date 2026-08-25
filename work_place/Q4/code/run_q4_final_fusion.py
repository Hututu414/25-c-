from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from run_q4_models import (
    DATA_DIR,
    OUTPUT_DIR,
    SEED,
    metric_record,
    patient_weights,
    sheet_spec,
)
from run_q4_round2 import calibration_statistics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

ROUND2_OOF = DATA_DIR / "q4_round2_oof_predictions.csv"
FIRST_OOF = DATA_DIR / "q4_oof_predictions.csv"
DIRECT_OOF = DATA_DIR / "q4_decision_layer_oof_predictions.csv"
FINAL_DIR = OUTPUT_DIR / "final_fusion"
PAYLOAD_PATH = OUTPUT_DIR / ".q4_final_fusion_payload.json"
FEATURES = ["p13", "p18", "p21"]
FUSIONS = {
    "Maximum risk": ("max", "score_max"),
    "Noisy-OR": ("noisy_or", "score_noisy_or"),
    "Logistic stacking": ("stack", "score_stack"),
}
CLASS_METRICS = [
    "Recall", "Specificity", "Precision", "F1", "F2", "Balanced Accuracy", "Accuracy"
]


def candidate_thresholds(probability: np.ndarray) -> np.ndarray:
    unique = np.unique(np.clip(probability, 0.0, 1.0))
    midpoints = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else np.array([])
    return np.unique(np.r_[0.0, 1.0, unique, midpoints])


def select_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    rule: str,
) -> dict[str, float]:
    best: tuple[tuple[float, ...], dict[str, float]] | None = None
    for threshold in candidate_thresholds(probability):
        prediction = probability >= threshold
        positive = weights[y == 1].sum()
        negative = weights[y == 0].sum()
        recall = float(weights[(y == 1) & prediction].sum() / positive)
        specificity = float(weights[(y == 0) & (~prediction)].sum() / negative)
        youden = recall + specificity - 1.0
        if rule == "Recall>=0.90" and recall < 0.90 - 1e-12:
            continue
        key = (
            (youden, specificity, recall, float(threshold))
            if rule == "Youden"
            else (specificity, recall, float(threshold))
        )
        row = {
            "Threshold": float(threshold),
            "Training Recall": recall,
            "Training Specificity": specificity,
            "Training Youden J": youden,
        }
        if best is None or key > best[0]:
            best = (key, row)
    if best is None:
        raise AssertionError(f"no feasible threshold for {rule}")
    return best[1]


def stacker() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", class_weight=None,
        max_iter=2000, random_state=SEED,
    )


def inner_crossfit_stack(training: pd.DataFrame) -> np.ndarray:
    probability = np.full(len(training), np.nan)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for train_pos, validation_pos in splitter.split(
        training, training["true_any"], groups=training["patient_id"]
    ):
        model = stacker()
        model.fit(training.iloc[train_pos][FEATURES], training.iloc[train_pos]["true_any"])
        probability[validation_pos] = model.predict_proba(training.iloc[validation_pos][FEATURES])[:, 1]
    if np.isnan(probability).any():
        raise AssertionError("inner cross-fit stacking probabilities are incomplete")
    return probability


def classification_row(
    rule: str,
    y: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | str]:
    metrics = metric_record(y, probability, prediction, weights)
    return {"Rule": rule, **{name: metrics[name] for name in CLASS_METRICS}}


def make_figures(frame: pd.DataFrame, youden: pd.DataFrame, weights: np.ndarray) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    colors = {
        "Maximum risk": "#0072B2",
        "Noisy-OR": "#D55E00",
        "Logistic stacking": "#009E73",
        "Direct Any M1-U": "#666666",
    }
    styles = {
        "Maximum risk": "-",
        "Noisy-OR": "--",
        "Logistic stacking": "-.",
        "Direct Any M1-U": ":",
    }
    y = frame["true_any"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(5.7, 4.1))
    curves = [(name, column) for name, (_, column) in FUSIONS.items()] + [("Direct Any M1-U", "pi_any")]
    for name, column in curves:
        precision, recall, _ = precision_recall_curve(y, frame[column], sample_weight=weights)
        ap = average_precision_score(y, frame[column], sample_weight=weights)
        ax.plot(
            recall, precision, color=colors[name], linestyle=styles[name], linewidth=2,
            label=f"{name} (AP={ap:.3f})",
        )
    ax.axhline(np.average(y, weights=weights), color="#999999", linestyle=(0, (1, 2)), label="Weighted prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "fusion_pr_curves.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    plotted = youden[youden["Rule"].isin([
        "Old: chromosome Minimax to hard OR",
        "Maximum risk + Youden",
        "Noisy-OR + Youden",
        "Logistic stacking + Youden",
    ])].copy()
    long = plotted.melt(id_vars="Rule", value_vars=["Recall", "Specificity", "Precision", "Balanced Accuracy"],
                        var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.barplot(
        data=long, x="Metric", y="Value", hue="Rule", ax=ax,
        palette=["#666666", "#0072B2", "#D55E00", "#009E73"], edgecolor="white",
    )
    ax.set(xlabel="", ylabel="Patient-weighted metric", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FINAL_DIR / "old_vs_soft_fusion_metrics.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def load_inputs() -> pd.DataFrame:
    round2 = pd.read_csv(ROUND2_OOF)
    first = pd.read_csv(FIRST_OOF)
    direct = pd.read_csv(DIRECT_OOF)
    for name, frame in (("round2", round2), ("first", first), ("direct", direct)):
        if not frame["row_id"].is_unique:
            raise AssertionError(f"{name} row_id is not unique")
    if not (set(round2["row_id"]) == set(first["row_id"]) == set(direct["row_id"])):
        raise AssertionError("Q4 OOF row_id sets differ")

    first_by_id = first.set_index("row_id")
    direct_by_id = direct.set_index("row_id")
    frame = round2[[
        "row_id", "patient_id", "fold", "true_T13", "true_T18", "true_T21", "true_any",
        "M1_U_prob_T13", "M1_U_prob_T18", "M1_U_prob_T21", "M1_U_prob_any",
        "M1_U_pred_any_MM",
    ]].copy()
    for column in ("patient_id", "fold", "true_any"):
        if not np.array_equal(frame[column].to_numpy(), frame["row_id"].map(direct_by_id[column]).to_numpy()):
            raise AssertionError(f"direct OOF mismatch: {column}")
    for column in ("patient_id", "fold", "true_any"):
        if not np.array_equal(frame[column].to_numpy(), frame["row_id"].map(first_by_id[column]).to_numpy()):
            raise AssertionError(f"first-round OOF mismatch: {column}")
    frame = frame.rename(columns={
        "M1_U_prob_T13": "p13", "M1_U_prob_T18": "p18", "M1_U_prob_T21": "p21",
        "M1_U_pred_any_MM": "old_hard_or",
    })
    frame["pi_any"] = frame["row_id"].map(direct_by_id["pi_any"])
    frame["direct_any_prediction"] = frame["row_id"].map(direct_by_id["binary_prediction"]).astype(int)
    frame["qc_low_confidence"] = frame["row_id"].map(first_by_id["qc_low_confidence"]).astype(bool)
    frame["score_max"] = frame[FEATURES].max(axis=1)
    frame["score_noisy_or"] = 1.0 - np.prod(1.0 - frame[FEATURES].to_numpy(float), axis=1)
    if not np.allclose(frame["score_noisy_or"], frame["M1_U_prob_any"], atol=1e-12):
        raise AssertionError("stored M1-U union probability is not the specified Noisy-OR")
    if frame.groupby("patient_id")["fold"].nunique().max() != 1:
        raise AssertionError("patient outer folds are not grouped")
    if len(frame) != 605 or frame["patient_id"].nunique() != 147:
        raise AssertionError("unexpected Q4 OOF cohort size")
    return frame


def main() -> None:
    np.random.seed(SEED)
    frame = load_inputs()
    threshold_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    frame["score_stack"] = np.nan

    for fold in sorted(frame["fold"].unique()):
        train_mask = frame["fold"].ne(fold)
        validation_mask = ~train_mask
        training = frame.loc[train_mask].copy()
        validation = frame.loc[validation_mask].copy()
        inner_stack = inner_crossfit_stack(training)
        model = stacker()
        model.fit(training[FEATURES], training["true_any"])
        frame.loc[validation_mask, "score_stack"] = model.predict_proba(validation[FEATURES])[:, 1]
        coefficient_rows.append({
            "fold": int(fold), "Intercept": float(model.intercept_[0]),
            "Coefficient p13": float(model.coef_[0, 0]),
            "Coefficient p18": float(model.coef_[0, 1]),
            "Coefficient p21": float(model.coef_[0, 2]),
            "Training rows": len(training), "Training patients": int(training["patient_id"].nunique()),
            "Model": "ordinary L2 logistic; C=1; no original features",
        })
        weights = patient_weights(training)
        y_train = training["true_any"].to_numpy(int)
        for fusion, (slug, score_column) in FUSIONS.items():
            training_score = inner_stack if fusion == "Logistic stacking" else training[score_column].to_numpy(float)
            for rule, suffix in (("Youden", "youden"), ("Recall>=0.90", "r90")):
                selected = select_threshold(y_train, training_score, weights, rule)
                threshold = selected["Threshold"]
                frame.loc[validation_mask, f"threshold_{slug}_{suffix}"] = threshold
                frame.loc[validation_mask, f"pred_{slug}_{suffix}"] = (
                    frame.loc[validation_mask, score_column].to_numpy(float) >= threshold
                ).astype(int)
                threshold_rows.append({
                    "fold": int(fold), "Fusion": fusion, "Threshold rule": rule,
                    **selected,
                    "Threshold source": (
                        "outer-training 3-fold cross-fit stack scores"
                        if fusion == "Logistic stacking"
                        else "strict outer-training chromosome OOF scores"
                    ),
                })

    required = ["score_stack"] + [
        f"{kind}_{slug}_{suffix}"
        for _, (slug, _) in FUSIONS.items()
        for kind in ("threshold", "pred")
        for suffix in ("youden", "r90")
    ]
    if frame[required].isna().any().any():
        raise AssertionError("strict OOF fusion outputs are incomplete")

    weights = patient_weights(frame)
    y = frame["true_any"].to_numpy(int)
    probability_rows = []
    for fusion, (slug, score_column) in FUSIONS.items():
        probability = frame[score_column].to_numpy(float)
        prediction = frame[f"pred_{slug}_youden"].to_numpy(int)
        metrics = metric_record(y, probability, prediction, weights)
        probability_rows.append({
            "Fusion": fusion,
            **{name: metrics[name] for name in ("PR-AUC", "ROC-AUC", "Brier")},
            **calibration_statistics(y, probability, weights),
        })
    probability_table = pd.DataFrame(probability_rows).sort_values(
        ["PR-AUC", "Brier", "ROC-AUC"], ascending=[False, True, False]
    ).reset_index(drop=True)
    probability_table.insert(0, "Rank", np.arange(1, len(probability_table) + 1))
    best_ap = float(probability_table["PR-AUC"].max())
    probability_table["Within 0.01 PR-AUC of best"] = probability_table["PR-AUC"].ge(best_ap - 0.01)
    complexity = {"Maximum risk": 0, "Noisy-OR": 1, "Logistic stacking": 2}
    eligible = probability_table[probability_table["Within 0.01 PR-AUC of best"]].copy()
    eligible["complexity"] = eligible["Fusion"].map(complexity)
    selected_fusion = eligible.sort_values(["complexity", "Brier", "ROC-AUC"], ascending=[True, True, False]).iloc[0]["Fusion"]
    probability_table["Selected by PR-AUC<0.01 simplicity rule"] = probability_table["Fusion"].eq(selected_fusion)

    direct_probability = frame["pi_any"].to_numpy(float)
    direct_metrics = metric_record(y, direct_probability, frame["direct_any_prediction"].to_numpy(int), weights)
    direct_calibration = calibration_statistics(y, direct_probability, weights)
    reference_probability = pd.DataFrame([
        {
            "Reference": "Direct Any M1-U", "PR-AUC": direct_metrics["PR-AUC"],
            "ROC-AUC": direct_metrics["ROC-AUC"], "Brier": direct_metrics["Brier"],
            "Calibration intercept": direct_calibration["Calibration intercept"],
            "Calibration slope": direct_calibration["Calibration slope"],
            "Note": "existing strict outer patient-level OOF reference",
        },
        {
            "Reference": "Old chromosome Minimax to hard OR", "PR-AUC": np.nan,
            "ROC-AUC": np.nan, "Brier": np.nan, "Calibration intercept": np.nan,
            "Calibration slope": np.nan,
            "Note": "hard OR has no natural continuous probability; classification metrics only",
        },
    ])

    youden_rows = [
        classification_row(
            "Old: chromosome Minimax to hard OR", y, frame["score_noisy_or"].to_numpy(float),
            frame["old_hard_or"].to_numpy(int), weights,
        ),
        classification_row(
            "Direct Any M1-U", y, direct_probability,
            frame["direct_any_prediction"].to_numpy(int), weights,
        ),
    ]
    high_rows = []
    threshold_summary_rows = []
    threshold_detail = pd.DataFrame(threshold_rows)
    for fusion, (slug, score_column) in FUSIONS.items():
        probability = frame[score_column].to_numpy(float)
        for rule, suffix in (("Youden", "youden"), ("Recall>=0.90", "r90")):
            prediction = frame[f"pred_{slug}_{suffix}"].to_numpy(int)
            row = classification_row(f"{fusion} + {rule}", y, probability, prediction, weights)
            selected_thresholds = threshold_detail[
                threshold_detail["Fusion"].eq(fusion) & threshold_detail["Threshold rule"].eq(rule)
            ]["Threshold"]
            row.update({
                "Fusion": fusion, "Threshold rule": rule,
                "Threshold": float(selected_thresholds.median()),
                "Threshold reporting": "median of five fold-specific training-only thresholds",
            })
            threshold_summary_rows.append(row)
            if rule == "Youden":
                youden_rows.append(classification_row(f"{fusion} + Youden", y, probability, prediction, weights))
            else:
                high_rows.append(row)
    youden_table = pd.DataFrame(youden_rows)
    high_table = pd.DataFrame(high_rows)
    threshold_summary = pd.DataFrame(threshold_summary_rows)[
        ["Fusion", "Threshold rule", "Threshold", *CLASS_METRICS, "Threshold reporting"]
    ]

    selected_slug, selected_score_column = FUSIONS[selected_fusion]
    selected_youden = youden_table.set_index("Rule").loc[f"{selected_fusion} + Youden"]
    final_suffix = "youden" if selected_youden["Recall"] >= 0.70 else "r90"
    final_rule = "Youden" if final_suffix == "youden" else "Recall>=0.90"
    frame["training_selected_threshold"] = frame[f"threshold_{selected_slug}_{final_suffix}"]
    frame["final_score"] = frame[selected_score_column]
    frame["pred_any"] = frame[f"pred_{selected_slug}_{final_suffix}"].astype(int)
    if not np.array_equal(
        frame["pred_any"].to_numpy(int),
        (frame["final_score"] >= frame["training_selected_threshold"]).astype(int).to_numpy(),
    ):
        raise AssertionError("final decision is inconsistent with its fold-specific threshold")

    final_metrics = metric_record(y, frame["final_score"], frame["pred_any"], weights)
    old_metrics = metric_record(y, frame["score_noisy_or"], frame["old_hard_or"], weights)
    freeze = bool(
        final_metrics["Recall"] >= 0.70
        and final_metrics["Specificity"] > old_metrics["Specificity"]
        and final_metrics["Precision"] > old_metrics["Precision"]
        and final_metrics["Balanced Accuracy"] > old_metrics["Balanced Accuracy"]
    )
    highest = frame[FEATURES].idxmax(axis=1).str.replace("p", "T", regex=False)
    frame["highest-risk chromosome"] = highest

    raw_columns = [
        "row_id", "patient_id", "fold", "true_any", "p13", "p18", "p21",
        "score_max", "score_noisy_or", "score_stack", "training_selected_threshold",
        "final_score", "pred_any", "qc_low_confidence",
    ]
    localization_columns = [
        "row_id", "patient_id", "fold", "true_any", "true_T13", "true_T18", "true_T21",
        "p13", "p18", "p21", "highest-risk chromosome", "qc_low_confidence",
    ]
    abnormal_localization = frame.loc[frame["pred_any"].eq(1), localization_columns].copy()
    final_summary = pd.DataFrame([
        {
            "Selected fusion": selected_fusion, "Final threshold rule": final_rule,
            "Threshold reporting": "fold-specific training-only; raw sheet stores each applied threshold",
            **{name: final_metrics[name] for name in ["PR-AUC", "ROC-AUC", "Brier", *CLASS_METRICS]},
            "Freeze Q4": freeze,
        }
    ])
    protocol = pd.DataFrame([
        {"Item": "Validation", "Definition": "strict outer patient-level OOF; 605 records from 147 patients"},
        {"Item": "Stacking", "Definition": "ordinary L2 logistic on p13/p18/p21 only; outer-fold cross-fit"},
        {"Item": "Threshold", "Definition": "selected only on outer-training OOF fusion scores"},
        {"Item": "QC", "Definition": "confidence flag only; does not change pred_any"},
        {"Item": "Localization", "Definition": "argmax(p13,p18,p21); T21 is exploratory because observations are very sparse"},
    ])

    selected_high = high_table.set_index("Fusion").loc[selected_fusion]
    recall_drop = float(old_metrics["Recall"] - final_metrics["Recall"])
    ranking_lines = "\n".join(
        f"| {int(rank)} | {fusion} | {ap:.6f} | {roc:.6f} | {brier:.6f} | {slope:.6f} |"
        for rank, fusion, ap, roc, brier, slope in probability_table[
            ["Rank", "Fusion", "PR-AUC", "ROC-AUC", "Brier", "Calibration slope"]
        ].itertuples(index=False, name=None)
    )
    youden_lines = "\n".join(
        f"| {rule} | {recall:.6f} | {specificity:.6f} | {precision:.6f} | {f1:.6f} | {f2:.6f} | {balanced:.6f} |"
        for rule, recall, specificity, precision, f1, f2, balanced in youden_table[
            ["Rule", "Recall", "Specificity", "Precision", "F1", "F2", "Balanced Accuracy"]
        ].itertuples(index=False, name=None)
    )
    limitation = "" if freeze else "\n当前数据的 AB 标签噪声与特征可分性限制了进一步提升。\n"
    summary = f"""# Q4 最终 Soft Fusion 摘要

## Fusion 排名

| Rank | Fusion | PR-AUC | ROC-AUC | Brier | Calibration slope |
| ---: | :--- | ---: | ---: | ---: | ---: |
{ranking_lines}

PR-AUC 差距小于 0.01 时按预设规则优先选择更简单、可解释的融合，因此最佳 fusion = {selected_fusion}。

## Youden 判定

| Rule | Recall | Specificity | Precision | F1 | F2 | Balanced Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
{youden_lines}

## 高敏感策略

最佳 fusion（{selected_fusion}）在 Recall>=90% 约束下：Recall={selected_high['Recall']:.6f}，Specificity={selected_high['Specificity']:.6f}，Precision={selected_high['Precision']:.6f}；所列阈值为五个 training-only outer-fold 阈值的中位数 {selected_high['Threshold']:.6f}。

## 最终推荐

最佳 fusion = {selected_fusion}

最终阈值规则 = {final_rule}（逐 outer fold 仅由 training OOF 分数选择）

是否显著改善旧 OR 的 Specificity = {'是' if final_metrics['Specificity'] > old_metrics['Specificity'] else '否'}（{old_metrics['Specificity']:.6f} -> {final_metrics['Specificity']:.6f}）

是否显著改善 Precision = {'是' if final_metrics['Precision'] > old_metrics['Precision'] else '否'}（{old_metrics['Precision']:.6f} -> {final_metrics['Precision']:.6f}）

Recall 下降多少 = {recall_drop:.6f}（{old_metrics['Recall']:.6f} -> {final_metrics['Recall']:.6f}；负值表示反而上升）

是否建议冻结 Q4 = {'是' if freeze else '否'}
{limitation}
"""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "q4_final_fusion_summary.md").write_text(summary, encoding="utf-8")

    payload = {
        "workbooks": [
            {
                "path": "final_fusion/fusion_probabilities.xlsx",
                "sheets": [
                    sheet_spec("OOF probabilities", frame[["row_id", "patient_id", "fold", "true_any", *FEATURES, "score_max", "score_noisy_or", "score_stack", "qc_low_confidence"]]),
                    sheet_spec("Stacker coefficients", pd.DataFrame(coefficient_rows)),
                    sheet_spec("Protocol", protocol),
                ],
            },
            {
                "path": "final_fusion/fusion_model_comparison.xlsx",
                "sheets": [
                    sheet_spec("Fusion ranking", probability_table),
                    sheet_spec("References", reference_probability),
                ],
            },
            {
                "path": "final_fusion/threshold_comparison.xlsx",
                "sheets": [
                    sheet_spec("Youden comparison", youden_table),
                    sheet_spec("High sensitivity", high_table),
                    sheet_spec("Threshold summary", threshold_summary),
                    sheet_spec("Thresholds by fold", threshold_detail),
                ],
            },
            {
                "path": "final_fusion/final_binary_decision.xlsx",
                "sheets": [
                    sheet_spec("OOF final decision", frame[raw_columns]),
                    sheet_spec("Final summary", final_summary),
                ],
            },
            {
                "path": "final_fusion/chromosome_risk_localization.xlsx",
                "sheets": [
                    sheet_spec("Abnormal localization", abnormal_localization),
                    sheet_spec("Protocol", protocol[protocol["Item"].isin(["QC", "Localization"])]),
                ],
            },
        ]
    }
    PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    make_figures(frame, youden_table, weights)
    print(f"selected_fusion={selected_fusion}")
    print(f"final_threshold_rule={final_rule}")
    print(f"old_recall={old_metrics['Recall']:.9f}")
    print(f"final_recall={final_metrics['Recall']:.9f}")
    print(f"old_specificity={old_metrics['Specificity']:.9f}")
    print(f"final_specificity={final_metrics['Specificity']:.9f}")
    print(f"old_precision={old_metrics['Precision']:.9f}")
    print(f"final_precision={final_metrics['Precision']:.9f}")
    print(f"freeze={freeze}")
    print(f"payload={PAYLOAD_PATH}")


if __name__ == "__main__":
    main()
