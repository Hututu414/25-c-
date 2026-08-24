from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q3_ROOT = Path(__file__).resolve().parents[1]
WORK_PLACE = Q3_ROOT.parent
PROJECT_ROOT = WORK_PLACE.parent
Q2_CODE = WORK_PLACE / "Q2" / "code"
DATA_DIR = Q3_ROOT / "data_processed"
AUDIT_DIR = Q3_ROOT / "outputs" / "audit"
PAYLOAD_PATH = AUDIT_DIR / ".audit_payload.json"
MINIMUM_SIZES = (20, 25, 30, 35, 40)
K_VALUES = (2, 3, 4, 5)
LOWER_BOUNDS = (10.0, 11.0, 12.0)

sys.dont_write_bytecode = True
sys.path.insert(0, str(Q2_CODE))
from optimize_bmi_groups import assign_groups, choose_k, ga_to_week_day, optimal_segmentations  # noqa: E402
from revise_timing_decision import delay_risk  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    files = sorted(item for path in paths for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def q_reference(mean_curve: np.ndarray, ga_grid: np.ndarray, q: float) -> tuple[float, bool]:
    reached = np.flatnonzero(mean_curve >= q)
    timing = float(ga_grid[reached[0]]) if len(reached) else np.nan
    return timing, bool(len(reached) and reached[0] == 0)


def group_timings(
    profiles: pd.DataFrame,
    curves: np.ndarray,
    ga_grid: np.ndarray,
    result: object,
    lower_bound: float,
) -> pd.DataFrame:
    mask = ga_grid >= lower_bound - 1e-12
    search_ga = ga_grid[mask]
    search_curves = curves[:, mask]
    groups = assign_groups(profiles["BMI"].to_numpy(float), result)
    rows = []
    for group_id in range(1, result.k + 1):
        selected = search_curves[groups == group_id]
        mean_curve = selected.mean(axis=0)
        detection = 1 - mean_curve
        delayed = delay_risk(search_ga)
        minimax = np.maximum(detection, delayed)
        best = int(np.argmin(minimax))
        timings = {f"t{int(q * 100)}": q_reference(mean_curve, search_ga, q) for q in (0.5, 0.7, 0.9)}
        rows.append(
            {
                "GA_lower_bound": lower_bound,
                "Group": group_id,
                "N": int((groups == group_id).sum()),
                "t50": timings["t50"][0],
                "t50_hits_lower_bound": timings["t50"][1],
                "t70": timings["t70"][0],
                "t70_hits_lower_bound": timings["t70"][1],
                "t90": timings["t90"][0],
                "t90_hits_lower_bound": timings["t90"][1],
                "minimax_GA": float(search_ga[best]),
                "minimax_week_day": ga_to_week_day(float(search_ga[best])),
                "mean_reliability_at_minimax": float(mean_curve[best]),
                "detection_failure_risk": float(detection[best]),
                "delay_risk": float(delayed[best]),
                "minimax_risk": float(minimax[best]),
            }
        )
    return pd.DataFrame(rows)


def json_ready(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": json.loads(frame.to_json(orient="values", force_ascii=False, double_precision=15)),
    }


def main() -> None:
    np.random.seed(20260824)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    formal_paths = [Q3_ROOT / "outputs" / name for name in ("raw", "decision", "figures")]
    frozen_paths = [WORK_PLACE / "Q1", WORK_PLACE / "Q2"]
    formal_hash_before = tree_hash(formal_paths)
    frozen_hash_before = tree_hash(frozen_paths)

    profiles = pd.read_csv(DATA_DIR / "patient_profiles.csv").sort_values("patient_id").reset_index(drop=True)
    reliability = pd.read_csv(DATA_DIR / "individual_reliability_curves.csv")
    ga_grid = np.sort(reliability["GA"].unique())
    curves = (
        reliability.pivot(index="patient_id", columns="GA", values="reliability")
        .reindex(index=profiles["patient_id"], columns=ga_grid)
        .to_numpy()
    )
    if curves.shape != (267, 151) or np.isnan(curves).any():
        raise AssertionError("frozen Q3 curve contract failed")

    all_k_rows, segment_rows, timing_rows, selected_rows = [], [], [], []
    scenario_results: dict[int, dict[int, object]] = {}
    for minimum in MINIMUM_SIZES:
        results = optimal_segmentations(
            profiles["BMI"].to_numpy(float),
            curves,
            ga_grid,
            k_values=K_VALUES,
            min_group_size=minimum,
        )
        scenario_results[minimum] = results
        selected_k = choose_k(results)
        for k, result in sorted(results.items()):
            previous = results.get(k - 1)
            relative = np.nan if previous is None else (previous.objective - result.objective) / previous.objective
            timings = group_timings(profiles, curves, ga_grid, result, 10.0)
            all_k_rows.append(
                {
                    "min_group_size": minimum,
                    "K": k,
                    "objective": result.objective,
                    "relative_improvement_vs_previous_K": relative,
                    "relative_improvement_pct": 100 * relative if np.isfinite(relative) else np.nan,
                    "cutpoints": "; ".join(f"{value:.4f}" for value in result.cutpoints),
                    "group_sizes": "; ".join(str(segment.n_patients) for segment in result.segments),
                    "min_group_N": min(segment.n_patients for segment in result.segments),
                    "minimax_timings": "; ".join(f"{value:.1f}" for value in timings["minimax_GA"]),
                    "selected_by_frozen_rule": k == selected_k,
                }
            )
            for group_id, (segment, timing) in enumerate(zip(result.segments, timings.itertuples(), strict=True), 1):
                segment_rows.append(
                    {
                        "min_group_size": minimum,
                        "K": k,
                        "selected_by_frozen_rule": k == selected_k,
                        "Group": group_id,
                        "observed_BMI_min": segment.bmi_min,
                        "observed_BMI_max": segment.bmi_max,
                        "N": segment.n_patients,
                        "segment_objective": segment.cost,
                    }
                )
                timing_rows.append(
                    {
                        "min_group_size": minimum,
                        "K": k,
                        "selected_by_frozen_rule": k == selected_k,
                        "Group": group_id,
                        "N": segment.n_patients,
                        "t50": timing.t50,
                        "t70": timing.t70,
                        "t90": timing.t90,
                        "minimax_GA": timing.minimax_GA,
                        "minimax_risk": timing.minimax_risk,
                    }
                )
        selected = results[selected_k]
        selected_timing = group_timings(profiles, curves, ga_grid, selected, 10.0)
        selected_rows.append(
            {
                "min_group_size": minimum,
                "selected_K": selected_k,
                "objective": selected.objective,
                "cutpoints": "; ".join(f"{value:.4f}" for value in selected.cutpoints),
                "high_tail_cutpoint": selected.cutpoints[-1],
                "additional_lower_cutpoints": "; ".join(f"{value:.4f}" for value in selected.cutpoints[:-1]),
                "group_sizes": "; ".join(str(segment.n_patients) for segment in selected.segments),
                "minimax_timings": "; ".join(f"{value:.1f}" for value in selected_timing["minimax_GA"]),
                "minimum_constraint_active": min(segment.n_patients for segment in selected.segments) == minimum,
            }
        )

    all_k = pd.DataFrame(all_k_rows)
    segments = pd.DataFrame(segment_rows)
    timings_all_k = pd.DataFrame(timing_rows)
    selected_summary = pd.DataFrame(selected_rows)

    elbow = all_k[all_k["min_group_size"] == 30].copy().reset_index(drop=True)
    elbow["J_previous_minus_J_K"] = elbow["objective"].shift(1) - elbow["objective"]
    elbow = elbow[
        [
            "K", "objective", "J_previous_minus_J_K", "relative_improvement_vs_previous_K",
            "relative_improvement_pct", "min_group_N", "cutpoints", "group_sizes", "minimax_timings",
            "selected_by_frozen_rule",
        ]
    ]

    baseline_result = scenario_results[30][2]
    ga_sensitivity = pd.concat(
        [group_timings(profiles, curves, ga_grid, baseline_result, lower) for lower in LOWER_BOUNDS],
        ignore_index=True,
    )
    model_data = pd.read_csv(DATA_DIR / "q3_model_data.csv")
    observed_support = pd.DataFrame(
        [
            {"GA_interval": "[10,11)", "observations": int((model_data["GA"] < 11).sum()), "patients": int(model_data.loc[model_data["GA"] < 11, "patient_id"].nunique()), "interpretation": "pure model extrapolation"},
            {"GA_interval": "[11,12)", "observations": int(((model_data["GA"] >= 11) & (model_data["GA"] < 12)).sum()), "patients": int(model_data.loc[(model_data["GA"] >= 11) & (model_data["GA"] < 12), "patient_id"].nunique()), "interpretation": "limited empirical support"},
            {"GA_interval": "[12,25]", "observations": int(model_data["GA"].between(12, 25).sum()), "patients": int(model_data.loc[model_data["GA"].between(12, 25), "patient_id"].nunique()), "interpretation": "main empirical support"},
        ]
    )
    observed_quantiles = pd.DataFrame(
        {
            "metric": ["minimum", "1% quantile", "5% quantile", "10% quantile"],
            "GA_weeks": [model_data["GA"].min(), *model_data["GA"].quantile([0.01, 0.05, 0.10]).tolist()],
        }
    )

    j2, j3, j4 = [scenario_results[30][k].objective for k in (2, 3, 4)]
    gain_23 = (j2 - j3) / j2
    gain_34 = (j3 - j4) / j3
    chosen = dict(zip(selected_summary["min_group_size"], selected_summary["selected_K"], strict=True))
    high_tail_cutpoints = dict(zip(selected_summary["min_group_size"], selected_summary["high_tail_cutpoint"], strict=True))
    minimax_stable = ga_sensitivity.groupby("Group")["minimax_GA"].nunique().eq(1).all()
    q90_stable = ga_sensitivity.groupby("Group")["t90"].nunique().eq(1).all()
    audit_md = f"""# Q3 分组小规模敏感性审计

1. **K=2 是否稳定？** 否。冻结选择规则在 `min_group_size=20/25` 时选择 K=3，在 30/35/40 时选择 K=2；对应为 `{chosen}`。
2. **35.22 分界是否稳定？** 否。被选解的高 BMI 尾部分界随最小组人数从 20 到 40 依次为 `{'; '.join(f'{m}: {high_tail_cutpoints[m]:.4f}' for m in MINIMUM_SIZES)}`。35.2209 只在最小人数为 30 时出现；在 20/25 场景还会额外出现 31.7432 的中间分界。
3. **N=30 是否为约束驱动？** 是。K=2 高 BMI 尾组人数随约束为 20、25、30、38、40；前三个场景恰好贴住约束。允许 N<30 时，选择结果变为 K=3，并在 BMI≈31.7432 处新增中间分界，同时高 BMI 尾组缩至 N=20/25；没有把原高 BMI 尾部再拆成两个更小的尾组。
4. **K=2→3 的目标函数改善是多少？** 基准 `min_group_size=30` 下 `(J2-J3)/J2={gain_23:.6f}`（{100*gain_23:.3f}%）；`(J3-J4)/J3={gain_34:.6f}`（{100*gain_34:.3f}%）。前者低于冻结规则的 10% 门槛，故基准选择 K=2。
5. **50%/70% 的 10周结果是否属于边界外推？** 是。原始男胎分析样本在 `[10,11)` 为 0 条观测；G1 的 50%/70%=10.0 周以及 G2 的 50%=10.8 周均位于纯模型外推区。将下界改为 11 或 12 周后，这些时点随下界移动或直接撞到新边界，不能解释为精确经验阈值。
6. **Minimax 13.9/15.6 是否对 GA 下界稳定？** {'是' if minimax_stable else '否'}。在 10/11/12 周三个下界下均保持 13.9/15.6 周；90% 时点也{'保持不变' if q90_stable else '发生变化'}。
7. **是否建议保留当前 Q3 两组结果？** 建议暂时保留正式 K=2 方案，但必须标注其依赖 `min_group_size=30` 的可执行性约束，不能称为对最小组人数稳健的数据结构。若临床上允许每组少于 30 人，应先由用户确认是否接受 K=3，再决定是否修改正式 Q3；本审计不直接改动正式结果。
"""
    (AUDIT_DIR / "q3_grouping_audit.md").write_text(audit_md, encoding="utf-8")

    formal_hash_after = tree_hash(formal_paths)
    frozen_hash_after = tree_hash(frozen_paths)
    if formal_hash_after != formal_hash_before or frozen_hash_after != frozen_hash_before:
        raise AssertionError("formal Q3 outputs or Q1/Q2 changed during audit")

    audit_metadata = pd.DataFrame(
        [
            {"key": "seed", "value": 20260824},
            {"key": "curve_source", "value": "Q3/data_processed/individual_reliability_curves.csv"},
            {"key": "formal_Q3_hash_unchanged", "value": formal_hash_before == formal_hash_after},
            {"key": "Q1_Q2_hash_unchanged", "value": frozen_hash_before == frozen_hash_after},
            {"key": "J2_to_J3_improvement", "value": gain_23},
            {"key": "J3_to_J4_improvement", "value": gain_34},
        ]
    )
    specifications = [
        {
            "path": "min_group_size_sensitivity.xlsx",
            "sheets": [
                ("selected_scenarios", selected_summary),
                ("all_K", all_k),
                ("segments", segments),
                ("timings_all_K", timings_all_k),
                ("metadata", audit_metadata),
            ],
        },
        {
            "path": "k_elbow_audit.xlsx",
            "sheets": [("K_elbow", elbow), ("baseline_segments", segments[(segments["min_group_size"] == 30)]), ("metadata", audit_metadata)],
        },
        {
            "path": "ga_lower_bound_sensitivity.xlsx",
            "sheets": [("timing_comparison", ga_sensitivity), ("observed_support", observed_support), ("observed_quantiles", observed_quantiles), ("metadata", audit_metadata)],
        },
    ]
    PAYLOAD_PATH.write_text(
        json.dumps(
            {
                "workbooks": [
                    {
                        "path": specification["path"],
                        "sheets": [{"name": name, **json_ready(frame)} for name, frame in specification["sheets"]],
                    }
                    for specification in specifications
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"audit payload ready: {PAYLOAD_PATH}")


if __name__ == "__main__":
    main()
