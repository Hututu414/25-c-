from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q3_ROOT = Path(__file__).resolve().parents[1]
WORK_PLACE = Q3_ROOT.parent
PROJECT_ROOT = WORK_PLACE.parent
Q2_CODE = WORK_PLACE / "Q2" / "code"
DATA_DIR = Q3_ROOT / "data_processed"
OUTPUT_DIR = Q3_ROOT / "outputs" / "audit" / "bootstrap_grouping"
PAYLOAD_PATH = OUTPUT_DIR / ".bootstrap_payload.json"

SEED = 20260824
B = 500
MINIMUM_SIZES = (20, 25)
K_VALUES = (2, 3, 4)

sys.dont_write_bytecode = True
sys.path.insert(0, str(Q2_CODE))
from optimize_bmi_groups import assign_groups, optimal_segmentations  # noqa: E402
from revise_timing_decision import delay_risk  # noqa: E402


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    files = sorted(item for root in paths for item in root.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash(item)))
    return digest.hexdigest()


def integration_weights(ga_grid: np.ndarray) -> np.ndarray:
    weights = np.full(len(ga_grid), float(np.diff(ga_grid).mean()))
    weights[[0, -1]] *= 0.5
    return weights


def training_group_curves(
    train_bmi: np.ndarray,
    train_curves: np.ndarray,
    result: object,
) -> np.ndarray:
    groups = assign_groups(train_bmi, result)
    return np.vstack([train_curves[groups == group].mean(axis=0) for group in range(1, result.k + 1)])


def minimax_timings(mean_curves: np.ndarray, ga_grid: np.ndarray) -> np.ndarray:
    delayed = delay_risk(ga_grid)
    return np.asarray(
        [ga_grid[int(np.argmin(np.maximum(1.0 - curve, delayed)))] for curve in mean_curves],
        dtype=float,
    )


def oob_error(
    oob_bmi: np.ndarray,
    oob_curves: np.ndarray,
    group_curves: np.ndarray,
    result: object,
    weights: np.ndarray,
) -> float:
    groups = assign_groups(oob_bmi, result) - 1
    residuals = oob_curves - group_curves[groups]
    return float(np.sum(residuals * residuals * weights))


def json_frame(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": json.loads(frame.to_json(orient="values", force_ascii=False, double_precision=15)),
    }


def summarize_k(raw: pd.DataFrame, minimum: int) -> tuple[pd.DataFrame, int]:
    selected = raw[raw["min_group_size"] == minimum]
    pivot = selected.pivot(index="bootstrap_id", columns="K", values="OOB_error")
    winner = pivot.idxmin(axis=1)
    rows = []
    for k in K_VALUES:
        values = pivot[k].to_numpy(float)
        mean = float(values.mean())
        standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
        rows.append(
            {
                "K": k,
                "Mean OOB Error": mean,
                "SE": standard_error,
                "95% CI Lower": mean - 1.96 * standard_error,
                "95% CI Upper": mean + 1.96 * standard_error,
                "Winner Frequency": float(np.mean(winner == k)),
            }
        )
    summary = pd.DataFrame(rows)
    best = summary.loc[summary["Mean OOB Error"].idxmin()]
    threshold = float(best["Mean OOB Error"] + best["SE"])
    summary["1-SE Eligible"] = summary["Mean OOB Error"] <= threshold + 1e-12
    recommended = int(summary.loc[summary["1-SE Eligible"], "K"].min())
    return summary, recommended


def summarize_cutpoints(
    k3: pd.DataFrame,
    minimum: int,
    full_cutpoints: tuple[float, float],
) -> pd.DataFrame:
    rows = []
    selected = k3[k3["min_group_size"] == minimum]
    for index, column in enumerate(("cutpoint_1", "cutpoint_2")):
        values = selected[column].to_numpy(float)
        q25, q75 = np.quantile(values, (0.25, 0.75))
        full = full_cutpoints[index]
        rows.append(
            {
                "cutpoint": f"c{index + 1}",
                "full_sample_cutpoint": full,
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "SD": float(values.std(ddof=1)),
                "IQR_lower": float(q25),
                "IQR_upper": float(q75),
                "IQR_width": float(q75 - q25),
                "2.5% quantile": float(np.quantile(values, 0.025)),
                "97.5% quantile": float(np.quantile(values, 0.975)),
                "proportion_within_0.5_BMI": float(np.mean(np.abs(values - full) <= 0.5 + 1e-12)),
                "proportion_within_1.0_BMI": float(np.mean(np.abs(values - full) <= 1.0 + 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def summarize_timings(
    k3: pd.DataFrame,
    minimum: int,
    full_timings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = k3[k3["min_group_size"] == minimum]
    rows = []
    timing_matrix = selected[["timing_1", "timing_2", "timing_3"]].to_numpy(float)
    for index in range(3):
        values = timing_matrix[:, index]
        full = float(full_timings[index])
        rows.append(
            {
                "Group": index + 1,
                "full_sample_timing": full,
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "SD": float(values.std(ddof=1)),
                "95% CI Lower": float(np.quantile(values, 0.025)),
                "95% CI Upper": float(np.quantile(values, 0.975)),
                "proportion_within_0.5_weeks": float(np.mean(np.abs(values - full) <= 0.5 + 1e-12)),
                "proportion_within_1.0_weeks": float(np.mean(np.abs(values - full) <= 1.0 + 1e-12)),
            }
        )
    monotone = np.all(np.diff(timing_matrix, axis=1) >= -1e-12, axis=1)
    check = pd.DataFrame(
        [
            {
                "min_group_size": minimum,
                "bootstrap_replicates": len(selected),
                "monotonic_non_decreasing_proportion": float(monotone.mean()),
            }
        ]
    )
    return pd.DataFrame(rows), check


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_q3 = [Q3_ROOT / "outputs" / name for name in ("raw", "decision", "figures")]
    frozen_q1_q2 = [WORK_PLACE / "Q1", WORK_PLACE / "Q2"]
    q3_hash_before = tree_hash(frozen_q3)
    q1_q2_hash_before = tree_hash(frozen_q1_q2)

    profiles = pd.read_csv(DATA_DIR / "patient_profiles.csv").sort_values("patient_id").reset_index(drop=True)
    reliability = pd.read_csv(DATA_DIR / "individual_reliability_curves.csv")
    ga_grid = np.sort(reliability["GA"].unique())
    curves = (
        reliability.pivot(index="patient_id", columns="GA", values="reliability")
        .reindex(index=profiles["patient_id"], columns=ga_grid)
        .to_numpy(float)
    )
    bmi = profiles["BMI"].to_numpy(float)
    if profiles["patient_id"].nunique() != 267 or curves.shape != (267, 151) or np.isnan(curves).any():
        raise AssertionError("frozen patient-level reliability curve contract failed")

    weights = integration_weights(ga_grid)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(profiles), size=(B, len(profiles)))

    full_results: dict[int, object] = {}
    full_timings: dict[int, np.ndarray] = {}
    for minimum in MINIMUM_SIZES:
        result = optimal_segmentations(bmi, curves, ga_grid, K_VALUES, minimum)[3]
        full_results[minimum] = result
        full_timings[minimum] = minimax_timings(training_group_curves(bmi, curves, result), ga_grid)

    rows: list[dict[str, object]] = []
    for minimum in MINIMUM_SIZES:
        for bootstrap_id, sample in enumerate(samples, 1):
            counts = np.bincount(sample, minlength=len(profiles))
            oob = counts == 0
            train_bmi, train_curves = bmi[sample], curves[sample]
            results = optimal_segmentations(train_bmi, train_curves, ga_grid, K_VALUES, minimum)
            for k in K_VALUES:
                result = results[k]
                group_curves = training_group_curves(train_bmi, train_curves, result)
                timings = minimax_timings(group_curves, ga_grid)
                cutpoints = list(result.cutpoints)
                sizes = [segment.n_patients for segment in result.segments]
                rows.append(
                    {
                        "bootstrap_id": bootstrap_id,
                        "min_group_size": minimum,
                        "K": k,
                        "objective": result.objective,
                        "OOB_error": oob_error(bmi[oob], curves[oob], group_curves, result, weights),
                        "OOB_N": int(oob.sum()),
                        "cutpoint_1": cutpoints[0],
                        "cutpoint_2": cutpoints[1] if len(cutpoints) > 1 else np.nan,
                        "cutpoint_3": cutpoints[2] if len(cutpoints) > 2 else np.nan,
                        "group_sizes": ";".join(map(str, sizes)),
                        "timing_1": timings[0],
                        "timing_2": timings[1],
                        "timing_3": timings[2] if len(timings) > 2 else np.nan,
                        "timing_4": timings[3] if len(timings) > 3 else np.nan,
                    }
                )
        print(f"bootstrap complete: min_group_size={minimum}, B={B}", flush=True)

    raw = pd.DataFrame(rows).sort_values(["min_group_size", "bootstrap_id", "K"]).reset_index(drop=True)
    if len(raw) != B * len(MINIMUM_SIZES) * len(K_VALUES):
        raise AssertionError("bootstrap row count mismatch")

    oob_summaries: dict[int, pd.DataFrame] = {}
    recommendations: dict[int, int] = {}
    cut_summaries: dict[int, pd.DataFrame] = {}
    timing_summaries: dict[int, pd.DataFrame] = {}
    monotonicity: dict[int, pd.DataFrame] = {}
    k3 = raw[raw["K"] == 3].copy()
    for minimum in MINIMUM_SIZES:
        oob_summaries[minimum], recommendations[minimum] = summarize_k(raw, minimum)
        full_cuts = tuple(float(value) for value in full_results[minimum].cutpoints)
        cut_summaries[minimum] = summarize_cutpoints(k3, minimum, full_cuts)
        timing_summaries[minimum], monotonicity[minimum] = summarize_timings(
            k3, minimum, full_timings[minimum]
        )

    main_oob = oob_summaries[20].set_index("K")
    main_cuts = cut_summaries[20].set_index("cutpoint")
    main_timing = timing_summaries[20].set_index("Group")
    if recommendations[20] == recommendations[25] == 3:
        verdict = "A. Bootstrap 明确支持 K=3，建议正式 Q3 改为三组；"
        full = full_results[20]
        suggested = (
            f"建议的全样本三组分界为 {full.cutpoints[0]:.4f}/{full.cutpoints[1]:.4f}，"
            f"Minimax 时点为 {'/'.join(f'{value:.1f}' for value in full_timings[20])} 周；等待用户确认后再修改正式结果。"
        )
    elif recommendations[20] == recommendations[25] == 2:
        verdict = "B. Bootstrap 明确支持 K=2，保留当前两组；"
        suggested = "主审计与轻量复核的 1-SE rule 均选择 K=2，因此本轮不建议修改正式 Q3。"
    else:
        verdict = "C. K=2/K=3 均无明显优势，只能稳定确认约 BMI=35 的高风险分界。"
        suggested = (
            f"主审计与轻量复核的 1-SE 推荐分别为 K={recommendations[20]}/K={recommendations[25]}，"
            "证据不足以修改正式 Q3。"
        )

    winner_text = ", ".join(
        f"K={k}: {main_oob.loc[k, 'Winner Frequency']:.1%}" for k in K_VALUES
    )
    summary = f"""# Q3 patient-level bootstrap 分组稳定性审计

### K 选择
```text
K=2 平均 OOB error = {main_oob.loc[2, 'Mean OOB Error']:.6f}
K=3 平均 OOB error = {main_oob.loc[3, 'Mean OOB Error']:.6f}
K=4 平均 OOB error = {main_oob.loc[4, 'Mean OOB Error']:.6f}

1-SE rule 推荐 K = {recommendations[20]}（min_group_size=25 复核：K={recommendations[25]}）
各 K 的 OOB winner frequency = {winner_text}
```

### K=3 稳定性
```text
c1 median / 95% CI = {main_cuts.loc['c1', 'median']:.4f} / [{main_cuts.loc['c1', '2.5% quantile']:.4f}, {main_cuts.loc['c1', '97.5% quantile']:.4f}]
c2 median / 95% CI = {main_cuts.loc['c2', 'median']:.4f} / [{main_cuts.loc['c2', '2.5% quantile']:.4f}, {main_cuts.loc['c2', '97.5% quantile']:.4f}]

c1 在全样本分界 ±1 BMI 内比例 = {main_cuts.loc['c1', 'proportion_within_1.0_BMI']:.1%}
c2 在全样本分界 ±1 BMI 内比例 = {main_cuts.loc['c2', 'proportion_within_1.0_BMI']:.1%}
```

### Minimax 时点
```text
G1 timing median / 95% CI = {main_timing.loc[1, 'median']:.1f} / [{main_timing.loc[1, '95% CI Lower']:.1f}, {main_timing.loc[1, '95% CI Upper']:.1f}] 周
G2 timing median / 95% CI = {main_timing.loc[2, 'median']:.1f} / [{main_timing.loc[2, '95% CI Lower']:.1f}, {main_timing.loc[2, '95% CI Upper']:.1f}] 周
G3 timing median / 95% CI = {main_timing.loc[3, 'median']:.1f} / [{main_timing.loc[3, '95% CI Lower']:.1f}, {main_timing.loc[3, '95% CI Upper']:.1f}] 周

三个时点保持单调不减的 bootstrap 比例 = {monotonicity[20].iloc[0]['monotonic_non_decreasing_proportion']:.1%}
```

### 最终判断

**{verdict}** {suggested}
"""
    (OUTPUT_DIR / "bootstrap_grouping_summary.md").write_text(summary, encoding="utf-8")

    q3_hash_after = tree_hash(frozen_q3)
    q1_q2_hash_after = tree_hash(frozen_q1_q2)
    if q3_hash_after != q3_hash_before or q1_q2_hash_after != q1_q2_hash_before:
        raise AssertionError("formal Q3 outputs or Q1/Q2 changed during bootstrap audit")

    metadata = pd.DataFrame(
        [
            {"key": "bootstrap_unit", "value": "patient_id"},
            {"key": "seed", "value": SEED},
            {"key": "B", "value": B},
            {"key": "patient_count", "value": len(profiles)},
            {"key": "K_values", "value": "2;3;4"},
            {"key": "min_group_sizes", "value": "20;25"},
            {"key": "OOB_error", "value": "sum of patient ISE using the frozen DP trapezoidal weights"},
            {"key": "SE", "value": "sample SD of 500 OOB errors divided by sqrt(B)"},
            {"key": "95% CI", "value": "mean OOB error plus or minus 1.96 SE"},
            {"key": "Q3_formal_outputs_unchanged", "value": q3_hash_before == q3_hash_after},
            {"key": "Q1_Q2_unchanged", "value": q1_q2_hash_before == q1_q2_hash_after},
        ]
    )
    payload = {
        "workbooks": [
            {
                "path": "bootstrap_raw.xlsx",
                "sheets": [
                    {"name": "bootstrap_raw", **json_frame(raw)},
                    {"name": "metadata", **json_frame(metadata)},
                ],
            },
            {
                "path": "k_oob_comparison.xlsx",
                "sheets": [
                    {"name": f"min_group_{minimum}", **json_frame(oob_summaries[minimum])}
                    for minimum in MINIMUM_SIZES
                ],
            },
            {
                "path": "k3_cutpoint_stability.xlsx",
                "sheets": [
                    {"name": f"min_group_{minimum}", **json_frame(cut_summaries[minimum])}
                    for minimum in MINIMUM_SIZES
                ],
            },
            {
                "path": "k3_timing_stability.xlsx",
                "sheets": [
                    *[
                        {"name": f"timing_min_{minimum}", **json_frame(timing_summaries[minimum])}
                        for minimum in MINIMUM_SIZES
                    ],
                    {
                        "name": "monotonicity",
                        **json_frame(pd.concat(monotonicity.values(), ignore_index=True)),
                    },
                ],
            },
        ]
    }
    PAYLOAD_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"bootstrap payload ready: {PAYLOAD_PATH}")


if __name__ == "__main__":
    main()
