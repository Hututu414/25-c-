from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optimize_bmi_groups import ga_to_week_day

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_processed"
DECISION_DIR = ROOT / "outputs" / "decision"
FIGURE_DIR = ROOT / "outputs" / "figures"
PAYLOAD_PATH = ROOT / "outputs" / ".decision_payload.json"

CUTPOINTS = np.array([32.0931884444288, 35.2208681140727])
INTERVALS = {1: "[20.70, 32.09]", 2: "(32.09, 35.22]", 3: "(35.22, 46.88]"}
EXPECTED_N = {1: 158, 2: 79, 3: 30}
SCENARIO_ORDER = ["no_error", "measurement_0_75", "measurement_1_00", "measurement_1_25"]
SCENARIO_LABEL = {
    "no_error": "无误差",
    "measurement_0_75": "0.75σ",
    "measurement_1_00": "1.00σ",
    "measurement_1_25": "1.25σ",
}


def delay_risk(ga: np.ndarray | pd.Series | float) -> np.ndarray:
    values = np.asarray(ga, dtype=float)
    return np.where(values <= 12, 0.0, np.where(values < 27, (values - 12) / 15, 1.0))


def assign_groups(bmi: pd.Series) -> np.ndarray:
    return np.searchsorted(CUTPOINTS, bmi.to_numpy(float), side="right") + 1


def validate_curve_source(frame: pd.DataFrame, scenarios: set[str]) -> None:
    required = {"patient_id", "BMI", "GA", "reliability"}
    if missing := required - set(frame.columns):
        raise ValueError(f"可靠性曲线缺少字段: {sorted(missing)}")
    if "scenario" in frame and set(frame["scenario"].unique()) != scenarios:
        raise ValueError("测量误差场景与既有四场景不一致")
    profile = frame[["patient_id", "BMI"]].drop_duplicates()
    if profile["patient_id"].duplicated().any() or len(profile) != 267:
        raise ValueError("可靠性曲线必须对应 267 位孕妇且每人只有一个 BMI")
    counts = pd.Series(assign_groups(profile["BMI"])).value_counts().sort_index().to_dict()
    if counts != EXPECTED_N:
        raise ValueError(f"固定 BMI 分组人数被改变: {counts}")
    ga = np.sort(frame["GA"].unique())
    expected_ga = np.round(np.arange(10.0, 25.0 + 0.05, 0.1), 1)
    if len(ga) != len(expected_ga) or not np.allclose(ga, expected_ga):
        raise ValueError("现有可靠性曲线不是 [10,25] 的 0.1 周网格")


def risk_curves(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    data = frame.copy()
    data["Group"] = assign_groups(data["BMI"])
    rows: list[pd.DataFrame] = []
    for group_id, selected in data.groupby("Group", sort=True):
        summary = (
            selected.groupby("GA", sort=True)["reliability"]
            .agg(
                mean_reliability="mean",
                q10_reliability=lambda values: values.quantile(0.10),
                min_reliability="min",
            )
            .reset_index()
        )
        summary.insert(0, "Group", int(group_id))
        summary.insert(0, "scenario", scenario)
        summary["BMI_interval"] = INTERVALS[int(group_id)]
        summary["N"] = EXPECTED_N[int(group_id)]
        summary["detection_failure_risk"] = 1 - summary["mean_reliability"]
        summary["delay_risk"] = delay_risk(summary["GA"])
        summary["minimax_risk"] = np.maximum(
            summary["detection_failure_risk"], summary["delay_risk"]
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def minimax_rows(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, group_id), selected in curves.groupby(["scenario", "Group"], sort=False):
        best = selected.loc[selected["minimax_risk"].idxmin()]
        rows.append(
            {
                "scenario": scenario,
                "Group": int(group_id),
                "BMI_interval": best["BMI_interval"],
                "N": int(best["N"]),
                "minimax_GA": float(best["GA"]),
                "week_day": ga_to_week_day(float(best["GA"])),
                "mean_reliability": float(best["mean_reliability"]),
                "q10_reliability": float(best["q10_reliability"]),
                "min_reliability": float(best["min_reliability"]),
                "detection_failure_risk": float(best["detection_failure_risk"]),
                "delay_risk": float(best["delay_risk"]),
                "minimax_risk": float(best["minimax_risk"]),
            }
        )
    return pd.DataFrame(rows)


def pareto_front(curve: pd.DataFrame) -> pd.DataFrame:
    ordered = curve.sort_values("GA").reset_index(drop=True)
    points = ordered[["delay_risk", "detection_failure_risk"]].to_numpy()
    is_pareto = np.array(
        [
            not np.any(np.all(points <= point, axis=1) & np.any(points < point, axis=1))
            for point in points
        ]
    )
    front = (
        ordered.loc[is_pareto]
        .sort_values(["delay_risk", "detection_failure_risk"])
        .drop_duplicates("delay_risk", keep="first")
        .reset_index(drop=True)
    )
    x = (front["delay_risk"] - front["delay_risk"].min()) / (
        front["delay_risk"].max() - front["delay_risk"].min()
    )
    y = (front["detection_failure_risk"] - front["detection_failure_risk"].min()) / (
        front["detection_failure_risk"].max() - front["detection_failure_risk"].min()
    )
    normalized = np.column_stack([x, y])
    chord = normalized[-1] - normalized[0]
    distances = np.abs(
        chord[0] * (normalized[:, 1] - normalized[0, 1])
        - chord[1] * (normalized[:, 0] - normalized[0, 0])
    ) / np.hypot(chord[0], chord[1])
    front["normalized_delay_risk"] = x
    front["normalized_detection_failure_risk"] = y
    front["knee_distance"] = distances
    front["is_knee"] = False
    front.loc[int(np.argmax(distances)), "is_knee"] = True
    return front


def dataframe_spec(name: str, frame: pd.DataFrame) -> dict[str, object]:
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return {"name": name, "columns": list(cleaned.columns), "rows": cleaned.values.tolist()}


def save_figure(curves: pd.DataFrame, final: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.4), sharex=True, sharey=True)
    for group_id, ax in enumerate(axes, 1):
        curve = curves[curves["Group"] == group_id]
        decision = final[final["组别"] == group_id].iloc[0]
        ax.plot(curve["GA"], curve["detection_failure_risk"], color="#0072B2", lw=2, label="检测失败风险")
        ax.plot(curve["GA"], curve["delay_risk"], color="#D55E00", lw=2, ls="--", label="延迟风险")
        ax.plot(curve["GA"], curve["minimax_risk"], color="#000000", lw=2.4, label="综合 Minimax 风险")
        ax.axvline(decision["最佳NIPT时点"], color="#CC79A7", lw=1.5, ls=":")
        ax.scatter(
            [decision["最佳NIPT时点"]],
            [decision["Minimax风险"]],
            color="#CC79A7",
            edgecolor="white",
            linewidth=0.8,
            s=44,
            zorder=5,
            label="Minimax 最优时点",
        )
        ax.set_title(f"G{group_id} | BMI {INTERVALS[group_id]} | N={EXPECTED_N[group_id]}")
        ax.set_xlabel("孕周（周）")
        ax.set_xlim(10, 25)
        ax.set_ylim(0, 0.9)
        ax.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.04,
            0.94,
            f"t*={decision['最佳NIPT时点']:.1f} 周\nR*={decision['Minimax风险']:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B7B7B7"},
        )
    axes[0].set_ylabel("归一化风险")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("NIPT 检测失败风险与延迟风险的 Minimax 权衡", y=1.10, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_group_timing.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(final: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    old_shift = final["原95%高保障时点"] - final["最佳NIPT时点"]
    max_error_shift = sensitivity["相对无误差时点变化"].abs().max()
    rows = "\n".join(
        f"| G{int(row['组别'])} | {row['BMI区间']} | {row['原95%高保障时点']:.1f} | "
        f"{row['最佳NIPT时点']:.1f} | {row['Pareto时点']:.1f} | {row['平均可靠性']:.4f} |"
        for _, row in final.iterrows()
    )
    tail = final.loc[final["最小可靠性"].idxmin()]
    summary = f"""# Q2 最佳 NIPT 时点修正

本次只复用既有可靠性曲线与固定 K=3 BMI 分组，未重新拟合 Q1、未重新计算可靠性曲线、未重新分组。

| Group | BMI区间 | 原95%高保障时点 | 新Minimax最佳时点 | Pareto knee | 平均可靠性 |
|---|---:|---:|---:|---:|---:|
{rows}

## 决策解释

- 新时点比原 95% 高保障时点提前 {old_shift.min():.1f}–{old_shift.max():.1f} 周，属于明显提前。
- 三组 Minimax 时点为 {', '.join(f'{value:.1f}' for value in final['最佳NIPT时点'])} 周，随 BMI 整体后移。
- Pareto knee 与 Minimax 的差值均不超过 1.0 周，支持主决策。knee 定义为两个风险分别归一化后，Pareto 前沿到端点连线的最大垂距点。
- 检测误差使推荐时点最多变化 {max_error_shift:.1f} 周。
- G{int(tail['组别'])} 组内尾部可靠性明显偏低：10% 分位为 {tail['10%分位可靠性']:.4f}，最小值为 {tail['最小可靠性']:.4f}。

原 q=90%、95%、97.5% 结果保留为“可靠性约束方案”，不再称为最终最佳时点。本轮未开展 Q3。
"""
    (DECISION_DIR / "q2_summary.md").write_text(summary, encoding="utf-8")


def self_check() -> None:
    assert np.allclose(delay_risk(np.array([10, 12, 13, 27, 30])), [0, 0, 1 / 15, 1, 1])
    assert assign_groups(
        pd.Series([20.70, CUTPOINTS[0], CUTPOINTS[0] + 0.001, CUTPOINTS[1] + 0.001])
    ).tolist() == [1, 2, 2, 3]


def main() -> None:
    self_check()
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    population = pd.read_csv(DATA_DIR / "population_curves.csv")
    measurement = pd.read_csv(DATA_DIR / "measurement_curves.csv")
    timing_all_q = pd.read_csv(DATA_DIR / "timing_all_q.csv")
    error_scenarios = pd.read_csv(DATA_DIR / "error_scenarios.csv")
    validate_curve_source(population, set())
    validate_curve_source(measurement, set(SCENARIO_ORDER))

    no_error_measurement = measurement[measurement["scenario"] == "no_error"].sort_values(["patient_id", "GA"])
    population_sorted = population.sort_values(["patient_id", "GA"])
    if not np.allclose(no_error_measurement["reliability"], population_sorted["reliability"], atol=1e-12):
        raise ValueError("测量误差文件中的无误差曲线与原可靠性曲线不一致")

    base_curves = risk_curves(population, "no_error")
    base_minimax = minimax_rows(base_curves).sort_values("Group").reset_index(drop=True)
    q95 = (
        timing_all_q[(timing_all_q["scenario"] == "no_error") & np.isclose(timing_all_q["q"], 0.95)]
        .set_index("Group")
        .sort_index()
    )

    knee_rows = []
    pareto_parts = []
    for group_id in EXPECTED_N:
        front = pareto_front(base_curves[base_curves["Group"] == group_id])
        knee = front.loc[front["is_knee"]].iloc[0]
        best = base_minimax[base_minimax["Group"] == group_id].iloc[0]
        pareto_parts.append(front)
        knee_rows.append(
            {
                "组别": group_id,
                "BMI区间": INTERVALS[group_id],
                "Minimax时点": best["minimax_GA"],
                "Pareto时点": float(knee["GA"]),
                "二者差值": abs(float(knee["GA"]) - best["minimax_GA"]),
                "稳定性判断": "支持（差值≤1周）" if abs(float(knee["GA"]) - best["minimax_GA"]) <= 1 else "不一致",
                "Pareto检测失败风险": float(knee["detection_failure_risk"]),
                "Pareto延迟风险": float(knee["delay_risk"]),
            }
        )
    pareto = pd.DataFrame(knee_rows)
    pareto_curve = pd.concat(pareto_parts, ignore_index=True)

    final = pd.DataFrame(
        {
            "组别": base_minimax["Group"],
            "BMI区间": base_minimax["BMI_interval"],
            "人数": base_minimax["N"],
            "原95%高保障时点": base_minimax["Group"].map(q95["recommended_GA"]),
            "最佳NIPT时点": base_minimax["minimax_GA"],
            "周+天": base_minimax["week_day"],
            "平均可靠性": base_minimax["mean_reliability"],
            "10%分位可靠性": base_minimax["q10_reliability"],
            "最小可靠性": base_minimax["min_reliability"],
            "检测失败风险": base_minimax["detection_failure_risk"],
            "延迟风险": base_minimax["delay_risk"],
            "Minimax风险": base_minimax["minimax_risk"],
            "Pareto时点": pareto["Pareto时点"],
            "Minimax与Pareto差值": pareto["二者差值"],
            "Pareto支持Minimax": pareto["稳定性判断"],
        }
    )

    scenario_curves = []
    for scenario in SCENARIO_ORDER:
        selected = measurement[measurement["scenario"] == scenario].drop(columns=["scenario", "measurement_sd"])
        scenario_curves.append(risk_curves(selected, scenario))
    measurement_risk = pd.concat(scenario_curves, ignore_index=True)
    sensitivity = minimax_rows(measurement_risk)
    sensitivity["场景"] = sensitivity["scenario"].map(SCENARIO_LABEL)
    sensitivity = sensitivity.merge(error_scenarios[["scenario", "measurement_sd"]], on="scenario", how="left")
    no_error_timing = sensitivity[sensitivity["scenario"] == "no_error"].set_index("Group")["minimax_GA"]
    sensitivity["相对无误差时点变化"] = sensitivity.apply(
        lambda row: row["minimax_GA"] - no_error_timing.loc[row["Group"]], axis=1
    )
    sensitivity["scenario_order"] = sensitivity["scenario"].map({name: i for i, name in enumerate(SCENARIO_ORDER)})
    sensitivity = sensitivity.sort_values(["scenario_order", "Group"]).reset_index(drop=True)
    sensitivity_output = sensitivity[
        [
            "场景",
            "measurement_sd",
            "Group",
            "BMI_interval",
            "N",
            "minimax_GA",
            "week_day",
            "mean_reliability",
            "q10_reliability",
            "min_reliability",
            "detection_failure_risk",
            "delay_risk",
            "minimax_risk",
            "相对无误差时点变化",
        ]
    ].rename(
        columns={
            "measurement_sd": "检测误差SD",
            "Group": "组别",
            "BMI_interval": "BMI区间",
            "N": "人数",
            "minimax_GA": "最佳NIPT时点",
            "week_day": "周+天",
            "mean_reliability": "平均可靠性",
            "q10_reliability": "10%分位可靠性",
            "min_reliability": "最小可靠性",
            "detection_failure_risk": "检测失败风险",
            "delay_risk": "延迟风险",
            "minimax_risk": "Minimax风险",
        }
    )
    scenario_summary = (
        sensitivity_output.groupby("场景", sort=False)
        .agg(
            最大绝对时点变化=("相对无误差时点变化", lambda values: values.abs().max()),
            最大Minimax风险=("Minimax风险", "max"),
        )
        .reset_index()
    )

    fixed_groups = final[["组别", "BMI区间", "人数"]].copy()
    fixed_groups["分组状态"] = "固定 K=3（本轮未重新分组）"

    high_assurance = timing_all_q[timing_all_q["scenario"] == "no_error"].copy()
    high_assurance.insert(0, "方案性质", "可靠性约束方案（不是最终最佳时点）")
    high_assurance = high_assurance[
        [
            "方案性质",
            "q",
            "Group",
            "BMI_interval",
            "N",
            "recommended_GA",
            "week_day",
            "mean_reliability",
            "q10_reliability",
            "min_reliability",
            "proportion_at_or_above_q",
        ]
    ].rename(
        columns={
            "q": "目标可靠性",
            "Group": "组别",
            "BMI_interval": "BMI区间",
            "N": "人数",
            "recommended_GA": "高保障时点",
            "week_day": "周+天",
            "mean_reliability": "平均可靠性",
            "q10_reliability": "10%分位可靠性",
            "min_reliability": "最小可靠性",
            "proportion_at_or_above_q": "达到目标的BMI比例",
        }
    )

    if final["人数"].tolist() != [158, 79, 30] or final["最佳NIPT时点"].tolist() != [13.9, 14.1, 15.4]:
        raise AssertionError("固定分组或 Minimax 决策回归检查失败")

    final.to_csv(DATA_DIR / "final_nipt.csv", index=False, encoding="utf-8-sig")
    base_curves.to_csv(DATA_DIR / "minimax_risk_curves.csv", index=False)
    pareto_curve.to_csv(DATA_DIR / "pareto_curve.csv", index=False)
    pareto.to_csv(DATA_DIR / "pareto_timing.csv", index=False, encoding="utf-8-sig")
    sensitivity_output.to_csv(DATA_DIR / "minimax_measurement_sensitivity.csv", index=False, encoding="utf-8-sig")
    high_assurance.to_csv(DATA_DIR / "high_assurance_timing.csv", index=False, encoding="utf-8-sig")
    save_figure(base_curves, final)
    write_summary(final, sensitivity_output)

    specifications = [
        {
            "path": "decision/final_nipt_timing.xlsx",
            "sheets": [dataframe_spec("final_minimax", final), dataframe_spec("pareto_review", pareto)],
        },
        {
            "path": "decision/sensitivity_summary.xlsx",
            "sheets": [
                dataframe_spec("measurement_minimax", sensitivity_output),
                dataframe_spec("scenario_summary", scenario_summary),
            ],
        },
        {
            "path": "decision/high_assurance_timing.xlsx",
            "sheets": [dataframe_spec("high_assurance", high_assurance)],
        },
        {
            "path": "decision/final_bmi_groups.xlsx",
            "sheets": [dataframe_spec("fixed_groups", fixed_groups)],
        },
    ]
    PAYLOAD_PATH.write_text(json.dumps({"workbooks": specifications}, ensure_ascii=False), encoding="utf-8")
    print("decision-only timing revision passed")


if __name__ == "__main__":
    main()
