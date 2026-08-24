from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q3_ROOT = Path(__file__).resolve().parents[1]
WORK_PLACE = Q3_ROOT.parent
Q2_CODE = WORK_PLACE / "Q2" / "code"
DATA_DIR = Q3_ROOT / "data_processed"
OUTPUT_DIR = Q3_ROOT / "outputs"
DECISION_DIR = OUTPUT_DIR / "decision"
PAYLOAD_PATH = OUTPUT_DIR / ".final_k3_payload.json"
RAW_MEASUREMENT = OUTPUT_DIR / "raw" / "q3_measurement_error_raw.xlsx"

FINAL_K = 3
MIN_GROUP_SIZE = 20

sys.dont_write_bytecode = True
sys.path.insert(0, str(Q2_CODE))
sys.path.insert(0, str(Q3_ROOT / "code"))
from optimize_bmi_groups import assign_groups, optimal_segmentations  # noqa: E402
from run_q3 import (  # noqa: E402
    group_decisions,
    individual_timing,
    interval_label,
    json_ready,
    q2_q3_comparison,
    save_figures,
    tree_hash,
    write_summary,
)


def main() -> None:
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    frozen_before = {name: tree_hash(WORK_PLACE / name) for name in ("Q1", "Q2")}

    profiles = pd.read_csv(DATA_DIR / "patient_profiles.csv").sort_values("patient_id").reset_index(drop=True)
    reliability = pd.read_csv(DATA_DIR / "individual_reliability_curves.csv")
    ga_grid = np.sort(reliability["GA"].unique())
    patient_ids = profiles["patient_id"].to_numpy()
    base_curves = (
        reliability.pivot(index="patient_id", columns="GA", values="reliability")
        .reindex(index=patient_ids, columns=ga_grid)
        .to_numpy(float)
    )
    if len(profiles) != 267 or base_curves.shape != (267, 151) or np.isnan(base_curves).any():
        raise AssertionError("frozen Q3 patient curve contract failed")

    final_result = optimal_segmentations(
        profiles["BMI"].to_numpy(float),
        base_curves,
        ga_grid,
        k_values=(FINAL_K,),
        min_group_size=MIN_GROUP_SIZE,
    )[FINAL_K]
    final_decisions, final_risks = group_decisions(
        profiles, base_curves, ga_grid, final_result, "no_error"
    )
    expected_cuts = np.array([31.7432307, 36.09730783])
    if not np.allclose(final_result.cutpoints, expected_cuts) or [segment.n_patients for segment in final_result.segments] != [146, 101, 20]:
        raise AssertionError("full-sample fixed-K=3 DP regression check failed")
    if not np.allclose(final_decisions["minimax_GA"], [13.8, 14.1, 16.1]):
        raise AssertionError("fixed-K=3 Minimax timing regression check failed")

    groups = assign_groups(profiles["BMI"].to_numpy(float), final_result)
    group_by_patient = dict(zip(patient_ids, groups, strict=True))
    individual_curves = reliability.drop(columns="Group", errors="ignore").copy()
    individual_curves["Group"] = individual_curves["patient_id"].map(group_by_patient)
    individual, loss_summary = individual_timing(
        profiles, base_curves, ga_grid, final_result, final_decisions
    )
    overview, comparison_long, comparison_shifts = q2_q3_comparison(
        profiles, final_decisions, final_result
    )

    sheet_names = {
        "no_error": "curves_no_error",
        "measurement_0_75": "curves_m0_75",
        "measurement_1_00": "curves_m1_00",
        "measurement_1_25": "curves_m1_25",
    }
    source = pd.read_excel(
        RAW_MEASUREMENT,
        sheet_name=["error_scenarios", *sheet_names.values()],
    )
    error_scenarios = source["error_scenarios"]
    measurement_sds = dict(
        zip(error_scenarios["scenario"], error_scenarios["measurement_sd"], strict=True)
    )
    sensitivity_parts = []
    error_segment_rows = []
    segmentation_parts = []
    segment_parts = []
    baseline_times = final_decisions.set_index("Group")["minimax_GA"]
    baseline_cuts = np.asarray(final_result.cutpoints)
    for scenario, sheet_name in sheet_names.items():
        frame = source[sheet_name]
        matrix = (
            frame.pivot(index="patient_id", columns="GA", values="reliability")
            .reindex(index=patient_ids, columns=ga_grid)
            .to_numpy(float)
        )
        if matrix.shape != (267, 151) or np.isnan(matrix).any():
            raise AssertionError(f"invalid frozen measurement curve matrix: {scenario}")
        if scenario == "no_error" and not np.allclose(matrix, base_curves, atol=1e-12):
            raise AssertionError("no-error raw workbook differs from frozen Q3 curves")
        result = optimal_segmentations(
            profiles["BMI"].to_numpy(float),
            matrix,
            ga_grid,
            k_values=(FINAL_K,),
            min_group_size=MIN_GROUP_SIZE,
        )[FINAL_K]
        decision, _ = group_decisions(profiles, matrix, ga_grid, result, scenario)
        cuts = np.asarray(result.cutpoints)
        decision["measurement_sd"] = measurement_sds[scenario]
        decision["cutpoints"] = "; ".join(f"{value:.4f}" for value in cuts)
        decision["max_cutpoint_shift_vs_no_error"] = float(np.max(np.abs(cuts - baseline_cuts)))
        decision["timing_shift_vs_no_error"] = decision.apply(
            lambda row: row["minimax_GA"] - baseline_times.loc[row["Group"]], axis=1
        )
        sensitivity_parts.append(decision)
        segmentation_parts.append(
            {
                "scenario": scenario,
                "K": FINAL_K,
                "total_curve_error": result.objective,
                "min_group_size": MIN_GROUP_SIZE,
                "min_group_N": min(segment.n_patients for segment in result.segments),
                "max_group_N": max(segment.n_patients for segment in result.segments),
                "cutpoints": "; ".join(f"{value:.4f}" for value in result.cutpoints),
                "selected": True,
                "selection_basis": "patient-level bootstrap OOB + 1-SE rule",
            }
        )
        for group_id, segment in enumerate(result.segments, 1):
            segment_parts.append(
                {
                    "scenario": scenario,
                    "K": FINAL_K,
                    "Group": group_id,
                    "BMI_interval": interval_label(result, group_id),
                    "observed_BMI_min": segment.bmi_min,
                    "observed_BMI_max": segment.bmi_max,
                    "N": segment.n_patients,
                    "group_curve_cost": segment.cost,
                }
            )
            error_segment_rows.append(
                {
                    "scenario": scenario,
                    "measurement_sd": measurement_sds[scenario],
                    "K": FINAL_K,
                    "Group": group_id,
                    "BMI_interval": interval_label(result, group_id),
                    "N": segment.n_patients,
                    "cutpoints": "; ".join(f"{value:.4f}" for value in cuts),
                    "group_curve_cost": segment.cost,
                }
            )

    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    error_segments = pd.DataFrame(error_segment_rows)
    segmentation_summary = pd.DataFrame(segmentation_parts)
    segments_all = pd.DataFrame(segment_parts)

    final_groups = final_decisions[
        ["Group", "BMI_interval", "N", "group_curve_cost", "minimax_GA", "week_day"]
    ].rename(
        columns={
            "Group": "组别",
            "BMI_interval": "BMI区间",
            "N": "孕妇数",
            "group_curve_cost": "组内曲线误差",
            "minimax_GA": "Minimax最佳时点",
            "week_day": "周+天",
        }
    )
    final_nipt = final_decisions.rename(
        columns={
            "Group": "组别",
            "BMI_interval": "BMI区间",
            "N": "人数",
            "t50": "50%可靠性参考时点",
            "t70": "70%可靠性参考时点",
            "t90": "90%可靠性参考时点",
            "minimax_GA": "Minimax最佳时点",
            "week_day": "周+天",
            "mean_reliability": "平均可靠性",
            "q10_reliability": "10%分位可靠性",
            "min_reliability": "最小可靠性",
            "detection_failure_risk": "检测失败风险",
            "delay_risk": "延迟风险",
            "minimax_risk": "Minimax风险",
        }
    )[
        [
            "组别",
            "BMI区间",
            "人数",
            "50%可靠性参考时点",
            "70%可靠性参考时点",
            "90%可靠性参考时点",
            "Minimax最佳时点",
            "周+天",
            "平均可靠性",
            "10%分位可靠性",
            "最小可靠性",
            "检测失败风险",
            "延迟风险",
            "Minimax风险",
        ]
    ]

    save_figures(
        profiles,
        base_curves,
        ga_grid,
        final_result,
        final_decisions,
        final_risks,
        comparison_long,
    )
    write_summary(
        final_decisions,
        final_result,
        overview,
        comparison_shifts,
        loss_summary,
        sensitivity,
        pd.read_csv(DATA_DIR / "model_smooth_terms.csv"),
        pd.read_csv(DATA_DIR / "model_k_check.csv"),
        pd.read_csv(DATA_DIR / "model_parametric_terms.csv"),
        pd.read_csv(DATA_DIR / "height_effect.csv"),
    )

    frames = {
        "individual_reliability_curves": individual_curves,
        "segmentation_summary": segmentation_summary,
        "segments_all": segments_all,
        "final_group_decisions": final_decisions,
        "final_risk_curves": final_risks,
        "individual_optimal_timing": individual,
        "individualization_loss": loss_summary,
        "q2_q3_overview": overview,
        "q2_q3_group_comparison": comparison_long,
        "q2_q3_shifts": comparison_shifts,
        "measurement_sensitivity": sensitivity,
        "measurement_error_segments": error_segments,
        "final_q3_bmi_groups": final_groups,
        "final_q3_nipt_timing": final_nipt,
    }
    for name, frame in frames.items():
        frame.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    frozen_after = {name: tree_hash(WORK_PLACE / name) for name in ("Q1", "Q2")}
    if frozen_after != frozen_before:
        raise AssertionError("Q1 or Q2 changed while freezing Q3")
    manifest_path = DATA_DIR / "q3_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "frozen_contracts_before": frozen_before,
            "frozen_contracts_after": frozen_after,
            "candidate_K": [FINAL_K],
            "selected_K": FINAL_K,
            "min_group_size": MIN_GROUP_SIZE,
            "K_selection_basis": "patient-level bootstrap OOB + 1-SE rule",
            "formal_BMI_cutpoints": list(final_result.cutpoints),
            "formal_group_sizes": [segment.n_patients for segment in final_result.segments],
            "formal_minimax_timings": final_decisions["minimax_GA"].tolist(),
            "bootstrap_stability": {
                "c1_within_full_plus_minus_1_BMI": 0.678,
                "c2_within_full_plus_minus_1_BMI": 0.986,
                "timing_monotonic_main": 1.0,
                "timing_monotonic_review": 0.998,
            },
            "freeze_status": "FINAL_Q3_K3",
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    reference_rows = [
        f"G{int(row.Group)}: {row.t50:.1f} / {row.t70:.1f} / {row.t90:.1f}"
        for row in final_decisions.itertuples()
    ]
    freeze = f"""Final K = 3
正式 BMI 分界 = {final_result.cutpoints[0]:.4f} / {final_result.cutpoints[1]:.4f}
三组人数 = {' / '.join(str(segment.n_patients) for segment in final_result.segments)}
50% / 70% / 90% 参考时点 = {'; '.join(reference_rows)}
三组 Minimax 最佳时点 = {' / '.join(f'{value:.1f}' for value in final_decisions['minimax_GA'])} 周
K 选择依据 = bootstrap OOB + 1-SE rule
高 BMI 分界稳定性结论 = c2 中位数 35.9718，95%区间 [35.3034, 36.4073]，98.6% 落在全样本分界 ±1 BMI 内，明显高于 c1 的 67.8%。
Minimax 时点稳定性结论 = 三组时点在主审计/复核中单调不减的比例为 100.0%/99.8%。

Q3 此后视为冻结版本，除发现代码错误或数据口径错误外，不再调整模型、组数、分界和风险函数。
"""
    (Q3_ROOT / "FINAL_Q3_FREEZE.md").write_text(freeze, encoding="utf-8")

    overview_display = overview.copy()
    overview_display["content"] = overview_display["content"].replace(
        {
            "K": "组数 K",
            "BMI cutpoints": "BMI 分界",
            "group N": "各组人数",
            "Minimax timing": "Minimax 最佳时点",
            "50% timing": "50%可靠性参考时点",
            "70% timing": "70%可靠性参考时点",
            "90% timing": "90%可靠性参考时点",
        }
    )
    comparison_display = comparison_long.rename(
        columns={
            "t50": "50%可靠性参考时点",
            "t70": "70%可靠性参考时点",
            "t90": "90%可靠性参考时点",
            "minimax_GA": "Minimax最佳时点",
        }
    )
    specifications = [
        {
            "path": "final_q3_bmi_groups.xlsx",
            "sheets": [{"name": "final_groups", **json_ready(final_groups)}],
        },
        {
            "path": "final_q3_nipt_timing.xlsx",
            "sheets": [{"name": "final_timing", **json_ready(final_nipt)}],
        },
        {
            "path": "q2_vs_q3_comparison.xlsx",
            "sheets": [
                {"name": "overview", **json_ready(overview_display)},
                {"name": "group_comparison", **json_ready(comparison_display)},
                {"name": "shifts", **json_ready(comparison_shifts)},
            ],
        },
        {
            "path": "individualization_loss.xlsx",
            "sheets": [
                {"name": "loss_summary", **json_ready(loss_summary)},
                {"name": "individual_timing", **json_ready(individual)},
            ],
        },
        {
            "path": "q3_sensitivity_summary.xlsx",
            "sheets": [
                {"name": "sensitivity", **json_ready(sensitivity)},
                {"name": "error_segments", **json_ready(error_segments)},
            ],
        },
    ]
    PAYLOAD_PATH.write_text(json.dumps({"workbooks": specifications}, ensure_ascii=False), encoding="utf-8")
    print(
        f"Q3 fixed-K=3 freeze payload ready: cuts={final_result.cutpoints}, "
        f"sizes={[segment.n_patients for segment in final_result.segments]}, "
        f"timings={final_decisions['minimax_GA'].tolist()}"
    )


if __name__ == "__main__":
    main()
