from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.polynomial.hermite import hermgauss

Q3_ROOT = Path(__file__).resolve().parents[1]
WORK_PLACE = Q3_ROOT.parent
PROJECT_ROOT = WORK_PLACE.parent
Q1_ROOT = WORK_PLACE / "Q1"
Q2_ROOT = WORK_PLACE / "Q2"
Q1_DATA = Q1_ROOT / "data_processed" / "q1_round2_model_data.csv"
SOURCE_XLSX = PROJECT_ROOT / "C题" / "附件.xlsx"
Q2_CODE = Q2_ROOT / "code"
DATA_DIR = Q3_ROOT / "data_processed"
RAW_DIR = Q3_ROOT / "outputs" / "raw"
DECISION_DIR = Q3_ROOT / "outputs" / "decision"
FIGURE_DIR = Q3_ROOT / "outputs" / "figures"
BOUNDARY_DIR = DATA_DIR / "q3_model_boundary"
PAYLOAD_PATH = Q3_ROOT / "outputs" / ".q3_workbook_payload.json"
SEED = 20260824
MIN_GROUP_SIZE = 20
FINAL_K = 3
Q_LEVELS = (0.50, 0.70, 0.90)

sys.path.insert(0, str(Q2_CODE))
from optimize_bmi_groups import (  # noqa: E402
    ProbabilityEngine,
    assign_groups,
    ga_to_week_day,
    optimal_segmentations,
)
from revise_timing_decision import delay_risk  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> dict[str, object]:
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and ".project-tmp" not in item.parts
    )
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return {"sha256": digest.hexdigest(), "files": len(files)}


def parse_ga(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)\s*[wW]\s*(?:\+\s*(\d+))?\s*", str(value))
    if not match:
        return np.nan
    weeks, days = int(match.group(1)), int(match.group(2) or 0)
    return weeks + days / 7 if 0 <= days <= 6 else np.nan


def prepare_q3_data() -> tuple[pd.DataFrame, dict[str, object]]:
    q1 = pd.read_csv(Q1_DATA)
    raw = pd.read_excel(SOURCE_XLSX, sheet_name=0, engine="openpyxl")
    patient_column, height_column = raw.columns[1], raw.columns[3]
    blood_column, ga_column = raw.columns[8], raw.columns[9]
    source = pd.DataFrame(
        {
            "raw_patient": raw[patient_column].astype(str),
            "blood_draw_no": pd.to_numeric(raw[blood_column], errors="coerce"),
            "GA": raw[ga_column].map(parse_ga),
            "Height": pd.to_numeric(raw[height_column], errors="coerce"),
        }
    )
    patients = sorted(source["raw_patient"].unique())
    patient_map = {value: f"P{index:04d}" for index, value in enumerate(patients, 1)}
    source["patient_id"] = source["raw_patient"].map(patient_map)
    if source[["patient_id", "blood_draw_no", "GA", "Height"]].isna().any().any():
        raise ValueError("source Height join contains missing or invalid fields")
    height = (
        source.groupby(["patient_id", "blood_draw_no", "GA"], as_index=False, sort=True)
        .agg(Height=("Height", "mean"))
    )
    result = q1.merge(height, on=["patient_id", "blood_draw_no", "GA"], how="left", validate="one_to_one")
    if len(result) != 1022 or result["patient_id"].nunique() != 267 or result["Height"].isna().any():
        raise AssertionError("Q3 Height augmentation changed the Q1 analytical sample")
    result.to_csv(DATA_DIR / "q3_model_data.csv", index=False, encoding="utf-8-sig")
    height_variation = source.groupby("patient_id")["Height"].nunique()
    audit = {
        "source_rows": len(raw),
        "q1_aggregated_rows": len(result),
        "patients": result["patient_id"].nunique(),
        "height_missing": int(result["Height"].isna().sum()),
        "patients_with_recorded_height_variation": int((height_variation > 1).sum()),
        "height_min": float(result["Height"].min()),
        "height_max": float(result["Height"].max()),
        "profile_definition": "earliest biological observation per patient; no future-patient information",
    }
    return result, audit


def find_rscript() -> Path:
    found = shutil.which("Rscript")
    if found:
        return Path(found)
    candidates = sorted((Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "R").glob("R-*/bin/Rscript.exe"), reverse=True)
    if not candidates:
        raise RuntimeError("Rscript not found")
    return candidates[0]


def run_r_model(r_temp: Path) -> dict[str, str]:
    if not r_temp.is_absolute() or any(ord(character) > 127 for character in str(r_temp)):
        raise ValueError("--r-temp must be an absolute writable ASCII path")
    r_temp.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix="q3_m4_", dir=r_temp))
    output = runtime / "output"
    adapter = runtime / "q3_reliability_from_m4.R"
    model_data = runtime / "q3_model_data.csv"
    shutil.copy2(Q3_ROOT / "code" / "q3_reliability_from_m4.R", adapter)
    shutil.copy2(DATA_DIR / "q3_model_data.csv", model_data)
    env = os.environ.copy()
    env.update({"TEMP": str(r_temp), "TMP": str(r_temp), "TMPDIR": str(r_temp), "LC_ALL": "C", "LANG": "C"})
    try:
        process = subprocess.run(
            [str(find_rscript()), "--vanilla", str(adapter), str(model_data), str(output)],
            cwd=runtime,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        stdout = process.stdout.decode("utf-8", errors="replace")
        stderr = process.stderr.decode("utf-8", errors="replace")
        error_file = output / "error.txt"
        if process.returncode:
            detail = error_file.read_text(encoding="utf-8", errors="replace") if error_file.exists() else stderr
            raise RuntimeError(f"Q3 R model failed: {detail}")
        if BOUNDARY_DIR.exists():
            shutil.rmtree(BOUNDARY_DIR)
        shutil.copytree(output, BOUNDARY_DIR)
        return {"stdout": stdout, "stderr": stderr}
    finally:
        shutil.rmtree(runtime, ignore_errors=True)


def read_model_info() -> dict[str, str]:
    frame = pd.read_csv(BOUNDARY_DIR / "model_info.csv", dtype=str, keep_default_na=False)
    return dict(zip(frame["key"], frame["value"], strict=True))


def interval_label(result: object, group_id: int) -> str:
    segment = result.segments[group_id - 1]
    lower = segment.bmi_min if group_id == 1 else result.cutpoints[group_id - 2]
    upper = segment.bmi_max if group_id == result.k else result.cutpoints[group_id - 1]
    return f"{'[' if group_id == 1 else '('}{lower:.2f}, {upper:.2f}]"


def segmentation_frames(results: dict[int, object], scenario: str, selected_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, segment_rows = [], []
    previous = None
    for k, result in sorted(results.items()):
        improvement = np.nan if previous is None else 100 * (previous - result.objective) / previous
        summary_rows.append(
            {
                "scenario": scenario,
                "K": k,
                "total_curve_error": result.objective,
                "marginal_improvement_pct": improvement,
                "min_group_N": min(segment.n_patients for segment in result.segments),
                "max_group_N": max(segment.n_patients for segment in result.segments),
                "cutpoints": "; ".join(f"{value:.4f}" for value in result.cutpoints),
                "selected": k == selected_k,
            }
        )
        for group_id, segment in enumerate(result.segments, 1):
            segment_rows.append(
                {
                    "scenario": scenario,
                    "K": k,
                    "Group": group_id,
                    "BMI_interval": interval_label(result, group_id),
                    "observed_BMI_min": segment.bmi_min,
                    "observed_BMI_max": segment.bmi_max,
                    "N": segment.n_patients,
                    "group_curve_cost": segment.cost,
                }
            )
        previous = result.objective
    return pd.DataFrame(summary_rows), pd.DataFrame(segment_rows)


def q_reference(mean_curve: np.ndarray, ga_grid: np.ndarray, q: float) -> float:
    reached = np.flatnonzero(mean_curve >= q)
    return float(ga_grid[reached[0]]) if len(reached) else np.nan


def group_decisions(
    profiles: pd.DataFrame,
    curves: np.ndarray,
    ga_grid: np.ndarray,
    result: object,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = assign_groups(profiles["BMI"].to_numpy(float), result)
    decision_rows, risk_rows = [], []
    delay = delay_risk(ga_grid)
    for group_id in range(1, result.k + 1):
        selected = curves[groups == group_id]
        mean_curve = selected.mean(axis=0)
        q10_curve = np.quantile(selected, 0.10, axis=0)
        min_curve = selected.min(axis=0)
        detection = 1 - mean_curve
        minimax = np.maximum(detection, delay)
        best_index = int(np.argmin(minimax))
        decision_rows.append(
            {
                "scenario": scenario,
                "Group": group_id,
                "BMI_interval": interval_label(result, group_id),
                "N": len(selected),
                "t50": q_reference(mean_curve, ga_grid, 0.50),
                "t70": q_reference(mean_curve, ga_grid, 0.70),
                "t90": q_reference(mean_curve, ga_grid, 0.90),
                "minimax_GA": float(ga_grid[best_index]),
                "week_day": ga_to_week_day(float(ga_grid[best_index])),
                "mean_reliability": float(mean_curve[best_index]),
                "q10_reliability": float(q10_curve[best_index]),
                "min_reliability": float(min_curve[best_index]),
                "detection_failure_risk": float(detection[best_index]),
                "delay_risk": float(delay[best_index]),
                "minimax_risk": float(minimax[best_index]),
                "group_curve_cost": result.segments[group_id - 1].cost,
            }
        )
        risk_rows.extend(
            {
                "scenario": scenario,
                "Group": group_id,
                "GA": float(ga),
                "mean_reliability": float(mean),
                "q10_reliability": float(q10),
                "min_reliability": float(minimum),
                "detection_failure_risk": float(det),
                "delay_risk": float(delayed),
                "minimax_risk": float(risk),
            }
            for ga, mean, q10, minimum, det, delayed, risk in zip(
                ga_grid, mean_curve, q10_curve, min_curve, detection, delay, minimax, strict=True
            )
        )
    return pd.DataFrame(decision_rows), pd.DataFrame(risk_rows)


def individual_timing(
    profiles: pd.DataFrame,
    curves: np.ndarray,
    ga_grid: np.ndarray,
    result: object,
    group_decision: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = assign_groups(profiles["BMI"].to_numpy(float), result)
    delay = delay_risk(ga_grid)
    group_times = group_decision.set_index("Group")["minimax_GA"]
    rows = []
    for index, (_, profile) in enumerate(profiles.iterrows()):
        individual_risk = np.maximum(1 - curves[index], delay)
        best = int(np.argmin(individual_risk))
        group_id = int(groups[index])
        group_time = float(group_times.loc[group_id])
        group_index = int(np.argmin(np.abs(ga_grid - group_time)))
        delta = float(ga_grid[best] - group_time)
        rows.append(
            {
                **profile.to_dict(),
                "Group": group_id,
                "individual_minimax_GA": float(ga_grid[best]),
                "individual_week_day": ga_to_week_day(float(ga_grid[best])),
                "group_minimax_GA": group_time,
                "group_week_day": ga_to_week_day(group_time),
                "delta_week": delta,
                "absolute_delta_week": abs(delta),
                "individual_minimax_risk": float(individual_risk[best]),
                "risk_at_group_time": float(individual_risk[group_index]),
                "excess_risk_at_group_time": float(individual_risk[group_index] - individual_risk[best]),
            }
        )
    individual = pd.DataFrame(rows)
    summary_rows = []
    for label, frame in [("Overall", individual), *[(f"G{g}", individual[individual["Group"] == g]) for g in sorted(individual["Group"].unique())]]:
        absolute = frame["absolute_delta_week"]
        summary_rows.append(
            {
                "scope": label,
                "N": len(frame),
                "mean_abs_delta": absolute.mean(),
                "median_abs_delta": absolute.median(),
                "q90_abs_delta": absolute.quantile(0.90),
                "max_abs_delta": absolute.max(),
                "proportion_abs_delta_le_0_5": (absolute <= 0.5 + 1e-12).mean(),
                "proportion_abs_delta_le_1_0": (absolute <= 1.0 + 1e-12).mean(),
                "proportion_abs_delta_le_2_0": (absolute <= 2.0 + 1e-12).mean(),
                "mean_excess_risk_at_group_time": frame["excess_risk_at_group_time"].mean(),
            }
        )
    return individual, pd.DataFrame(summary_rows)


def q2_reference() -> tuple[pd.DataFrame, pd.DataFrame, int, tuple[float, ...]]:
    segments = pd.read_csv(Q2_ROOT / "data_processed" / "segments_all.csv")
    selected = pd.read_csv(Q2_ROOT / "data_processed" / "segmentation_summary.csv")
    selected_k = int(selected[(selected["scenario"] == "no_error") & selected["selected"]]["K"].iloc[0])
    q2_segments = segments[(segments["scenario"] == "no_error") & (segments["K"] == selected_k)].sort_values("Group")
    cutpoints = tuple(float(q2_segments.iloc[index]["BMI_upper"]) for index in range(selected_k - 1))
    profiles = pd.read_csv(Q2_ROOT / "data_processed" / "patient_profiles.csv").sort_values("patient_id")
    curves = pd.read_csv(Q2_ROOT / "data_processed" / "population_curves.csv")
    ga_grid = np.sort(curves["GA"].unique())
    matrix = (
        curves.pivot(index="patient_id", columns="GA", values="reliability")
        .reindex(index=profiles["patient_id"], columns=ga_grid)
        .to_numpy()
    )
    groups = np.searchsorted(np.asarray(cutpoints), profiles["BMI"].to_numpy(float), side="right") + 1
    minimax = pd.read_csv(Q2_ROOT / "data_processed" / "final_nipt.csv").sort_values("组别")
    rows = []
    for group_id in range(1, selected_k + 1):
        selected_curves = matrix[groups == group_id]
        mean_curve = selected_curves.mean(axis=0)
        row = minimax[minimax["组别"] == group_id].iloc[0]
        rows.append(
            {
                "model": "Q2",
                "Group": group_id,
                "BMI_interval": row["BMI区间"],
                "BMI_midpoint": float(profiles.loc[groups == group_id, "BMI"].mean()),
                "N": int((groups == group_id).sum()),
                "t50": q_reference(mean_curve, ga_grid, 0.50),
                "t70": q_reference(mean_curve, ga_grid, 0.70),
                "t90": q_reference(mean_curve, ga_grid, 0.90),
                "minimax_GA": float(row["最佳NIPT时点"]),
            }
        )
    return pd.DataFrame(rows), q2_segments, selected_k, cutpoints


def q2_q3_comparison(
    profiles: pd.DataFrame, q3: pd.DataFrame, q3_result: object
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q2, q2_segments, q2_k, q2_cuts = q2_reference()
    q3_long = q3[["Group", "BMI_interval", "N", "t50", "t70", "t90", "minimax_GA"]].copy()
    q3_long.insert(0, "model", "Q3")
    q3_groups = assign_groups(profiles["BMI"].to_numpy(float), q3_result)
    q3_long.insert(
        3,
        "BMI_midpoint",
        [float(profiles.loc[q3_groups == group_id, "BMI"].mean()) for group_id in q3_long["Group"]],
    )
    long = pd.concat([q2, q3_long], ignore_index=True)
    overview = pd.DataFrame(
        [
            {
                "content": "K",
                "Q2": str(q2_k),
                "Q3": str(q3_result.k),
                "change": str(q3_result.k - q2_k),
            },
            {
                "content": "BMI cutpoints",
                "Q2": "; ".join(f"{value:.4f}" for value in q2_cuts),
                "Q3": "; ".join(f"{value:.4f}" for value in q3_result.cutpoints),
                "change": "group-wise details in shift table",
            },
            {
                "content": "group N",
                "Q2": "; ".join(str(value) for value in q2["N"]),
                "Q3": "; ".join(str(value) for value in q3["N"]),
                "change": "group-wise details in shift table",
            },
            {
                "content": "Minimax timing",
                "Q2": "; ".join(f"{value:.1f}" for value in q2["minimax_GA"]),
                "Q3": "; ".join(f"{value:.1f}" for value in q3["minimax_GA"]),
                "change": "group-wise details in shift table",
            },
            {
                "content": "50% timing",
                "Q2": "; ".join(f"{value:.1f}" for value in q2["t50"]),
                "Q3": "; ".join(f"{value:.1f}" for value in q3["t50"]),
                "change": "group-wise details in shift table",
            },
            {
                "content": "70% timing",
                "Q2": "; ".join(f"{value:.1f}" for value in q2["t70"]),
                "Q3": "; ".join(f"{value:.1f}" for value in q3["t70"]),
                "change": "group-wise details in shift table",
            },
            {
                "content": "90% timing",
                "Q2": "; ".join(f"{value:.1f}" for value in q2["t90"]),
                "Q3": "; ".join(f"{value:.1f}" for value in q3["t90"]),
                "change": "group-wise details in shift table",
            },
        ]
    )
    cut_rows = []
    if len(q2_cuts) == len(q3_result.cutpoints):
        for index, (before, after) in enumerate(zip(q2_cuts, q3_result.cutpoints, strict=True), 1):
            cut_rows.append(
                {
                    "record_type": "cutpoint",
                    "status": "matched_by_order",
                    "Q2_cutpoint": index,
                    "Q3_cutpoint": index,
                    "Q2_BMI_cutpoint": before,
                    "Q3_BMI_cutpoint": after,
                    "BMI_cutpoint_shift": after - before,
                }
            )
    else:
        matched_q3: set[int] = set()
        for q2_index, before in enumerate(q2_cuts, 1):
            differences = np.abs(np.asarray(q3_result.cutpoints) - before)
            nearest = int(np.argmin(differences)) if len(differences) else -1
            retained = nearest >= 0 and differences[nearest] < 1e-8
            if retained:
                matched_q3.add(nearest)
            cut_rows.append(
                {
                    "record_type": "cutpoint",
                    "status": "retained" if retained else "removed",
                    "Q2_cutpoint": q2_index,
                    "Q3_cutpoint": nearest + 1 if retained else np.nan,
                    "Q2_BMI_cutpoint": before,
                    "Q3_BMI_cutpoint": q3_result.cutpoints[nearest] if retained else np.nan,
                    "BMI_cutpoint_shift": q3_result.cutpoints[nearest] - before if retained else np.nan,
                }
            )
        for q3_index, after in enumerate(q3_result.cutpoints):
            if q3_index not in matched_q3:
                cut_rows.append(
                    {
                        "record_type": "cutpoint",
                        "status": "added",
                        "Q2_cutpoint": np.nan,
                        "Q3_cutpoint": q3_index + 1,
                        "Q2_BMI_cutpoint": np.nan,
                        "Q3_BMI_cutpoint": after,
                        "BMI_cutpoint_shift": np.nan,
                    }
                )
    mapping_rows = []
    for q3_group, q3_segment in enumerate(q3_result.segments, 1):
        q3_row = q3[q3["Group"] == q3_group].iloc[0]
        for _, q2_segment in q2_segments.iterrows():
            overlap = max(
                0.0,
                min(q3_segment.bmi_max, q2_segment["observed_BMI_max"])
                - max(q3_segment.bmi_min, q2_segment["observed_BMI_min"]),
            )
            if overlap <= 0 and not math.isclose(q3_segment.bmi_min, q2_segment["observed_BMI_max"]):
                continue
            q2_group = int(q2_segment["Group"])
            q2_row = q2[q2["Group"] == q2_group].iloc[0]
            row = {
                "record_type": "group_mapping",
                "status": "matched_by_BMI_overlap",
                "Q2_Group": q2_group,
                "Q3_Group": q3_group,
                "Q2_BMI_interval": q2_row["BMI_interval"],
                "Q3_BMI_interval": q3_row["BMI_interval"],
                "Q2_N": q2_row["N"],
                "Q3_N": q3_row["N"],
            }
            for metric in ["t50", "t70", "t90", "minimax_GA"]:
                row[f"Q2_{metric}"] = q2_row[metric]
                row[f"Q3_{metric}"] = q3_row[metric]
                row[f"shift_{metric}"] = q3_row[metric] - q2_row[metric]
            mapping_rows.append(row)
    return overview, long, pd.concat([pd.DataFrame(cut_rows), pd.DataFrame(mapping_rows)], ignore_index=True, sort=False)


def configure_figures() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def save_figures(
    profiles: pd.DataFrame,
    curves: np.ndarray,
    ga_grid: np.ndarray,
    result: object,
    decisions: pd.DataFrame,
    risks: pd.DataFrame,
    comparison_long: pd.DataFrame,
) -> None:
    configure_figures()
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    q10, median, q90 = np.quantile(curves, [0.10, 0.50, 0.90], axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for curve in curves[::3]:
        ax.plot(ga_grid, curve, color="#8C8C8C", alpha=0.11, lw=0.7)
    ax.fill_between(ga_grid, q10, q90, color=colors[0], alpha=0.18, label="10%–90% 分位带")
    ax.plot(ga_grid, median, color=colors[0], lw=2.3, label="个体曲线中位数")
    ax.axhline(0.90, color="#555555", lw=1, ls="--")
    ax.set(xlim=(10, 25), ylim=(0, 1.01), xlabel="孕周（周）", ylabel="达标可靠性", title="多因素条件下个体可靠性曲线的异质性")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q3_individual_reliability_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    groups = assign_groups(profiles["BMI"].to_numpy(float), result)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    axes[0].hist(profiles["BMI"], bins=24, color="#B9CDE5", edgecolor="white")
    for cutpoint in result.cutpoints:
        axes[0].axvline(cutpoint, color="#D55E00", lw=1.8, ls="--")
    axes[0].set(xlabel="首次观测 BMI", ylabel="孕妇数", title=f"Q3 连续 BMI 分组（K={result.k}）")
    for group_id in range(1, result.k + 1):
        mean_curve = curves[groups == group_id].mean(axis=0)
        axes[1].plot(
            ga_grid,
            mean_curve,
            color=colors[(group_id - 1) % len(colors)],
            lw=2,
            label=f"G{group_id}: {interval_label(result, group_id)} (N={int((groups == group_id).sum())})",
        )
    axes[1].set(xlim=(10, 25), ylim=(0, 1.01), xlabel="孕周（周）", ylabel="组平均可靠性", title="最终各组平均可靠性曲线")
    axes[1].legend(frameon=False)
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q3_bmi_groups.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, result.k, figsize=(3.5 * result.k, 4.0), sharex=True, sharey=True, squeeze=False)
    for group_id, ax in enumerate(axes[0], 1):
        frame = risks[risks["Group"] == group_id]
        decision = decisions[decisions["Group"] == group_id].iloc[0]
        ax.plot(frame["GA"], frame["detection_failure_risk"], color=colors[0], lw=2, label="检测失败风险")
        ax.plot(frame["GA"], frame["delay_risk"], color=colors[1], lw=2, ls="--", label="延迟风险")
        ax.plot(frame["GA"], frame["minimax_risk"], color="#000000", lw=2.2, label="Minimax 风险")
        ax.axvline(decision["minimax_GA"], color=colors[3], lw=1.5, ls=":")
        ax.scatter([decision["minimax_GA"]], [decision["minimax_risk"]], color=colors[3], s=38, zorder=4)
        ax.set(
            xlim=(10, 25),
            ylim=(0, 0.90),
            xlabel="孕周（周）",
            title=f"G{group_id} | N={int(decision['N'])} | t*={decision['minimax_GA']:.1f}",
        )
        sns.despine(ax=ax)
    axes[0, 0].set_ylabel("归一化风险")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q3_minimax_timing.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for model, marker, color in [("Q2", "o", colors[0]), ("Q3", "s", colors[1])]:
        frame = comparison_long[comparison_long["model"] == model]
        axes[0].plot(frame["BMI_midpoint"], frame["minimax_GA"], marker=marker, color=color, lw=2, label=model)
        for metric, ls in [("t50", ":"), ("t70", "-."), ("t90", "--")]:
            axes[1].plot(frame["BMI_midpoint"], frame[metric], marker=marker, color=color, lw=1.5, ls=ls, label=f"{model} {metric[1:]}%")
    axes[0].set(xlabel="组内平均 BMI", ylabel="孕周（周）", title="Minimax 时点：Q2 vs Q3")
    axes[1].set(xlabel="组内平均 BMI", ylabel="孕周（周）", title="可靠性参考时点：Q2 vs Q3")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, ncol=2)
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "q2_vs_q3_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def json_ready(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": json.loads(frame.to_json(orient="values", force_ascii=False, double_precision=15)),
    }


def build_payload(specifications: list[dict[str, object]]) -> None:
    PAYLOAD_PATH.write_text(json.dumps({"workbooks": specifications}, ensure_ascii=False), encoding="utf-8")


def write_summary(
    decisions: pd.DataFrame,
    result: object,
    overview: pd.DataFrame,
    shifts: pd.DataFrame,
    loss_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    smooth: pd.DataFrame,
    k_check: pd.DataFrame,
    parametric: pd.DataFrame,
    height_effect: pd.DataFrame,
) -> None:
    height_row = smooth[smooth["smooth_term"] == "s(Height_c)"].iloc[0]
    height_k = k_check[k_check["smooth_term"] == "s(Height_c)"].iloc[0]
    p_column = next(column for column in smooth.columns if "p-value" in column)
    k_column = next(column for column in k_check.columns if "k-index" in column)
    rho = height_effect[["Height", "link_partial_effect"]].corr(method="spearman").iloc[0, 1]
    height_direction = "总体上升" if rho > 0.7 else "总体下降" if rho < -0.7 else "非单调"
    estimate_column = "Estimate"
    param_p_column = next(column for column in parametric.columns if column.startswith("Pr("))
    clinical_lines = []
    for term in ["AGE_c", "gravidity_cat2", "gravidity_cat3plus", "parity", "conception_modeassisted"]:
        selected = parametric[parametric["term"] == term]
        if len(selected):
            row = selected.iloc[0]
            direction = "正向" if row[estimate_column] > 0 else "负向"
            significance = "显著" if row[param_p_column] < 0.05 else "不显著"
            clinical_lines.append(f"- `{term}`：{direction}，p={row[param_p_column]:.4g}（{significance}）。")
    table_rows = "\n".join(
        f"| G{int(row.Group)} | {row.BMI_interval} | {int(row.N)} | {row.t50:.1f} | {row.t70:.1f} | {row.t90:.1f} | {row.minimax_GA:.1f} | {row.mean_reliability:.4f} | {row.q10_reliability:.4f} |"
        for row in decisions.itertuples()
    )
    loss = loss_summary[loss_summary["scope"] == "Overall"].iloc[0]
    error = sensitivity[sensitivity["scenario"] != "no_error"]
    max_time_shift = error["timing_shift_vs_no_error"].abs().max()
    max_cut_shift = error["max_cutpoint_shift_vs_no_error"].max()
    group_shift = shifts[shifts["record_type"] == "group_mapping"]
    max_q2_q3_timing = group_shift["shift_minimax_GA"].abs().max()
    matched_cuts = shifts[(shifts["record_type"] == "cutpoint") & shifts["BMI_cutpoint_shift"].notna()]
    cut_shift = matched_cuts["BMI_cutpoint_shift"].abs().max() if len(matched_cuts) else np.nan
    removed_cuts = shifts[(shifts["record_type"] == "cutpoint") & (shifts["status"] == "removed")]["Q2_BMI_cutpoint"]
    boundary_text = (
        f"保留边界最大移动 {cut_shift:.4f}；删除 Q2 边界 {', '.join(f'{value:.4f}' for value in removed_cuts)}"
        if len(removed_cuts)
        else f"对应边界最大移动 {cut_shift:.4f}"
    )
    tail = decisions.loc[decisions["min_reliability"].idxmin()]
    summary = f"""# Q3 多因素个体可靠性与可执行 BMI 分组

Q1/Q2 均作为冻结输入读取；Q3 仅在原 Beta-GAMM 中新增 `s(Height_c, k=4)`，其余模型定义、测量误差模型、动态规划和 Minimax 风险函数均直接沿用。

| Group | BMI区间 | N | 50%可靠性参考时点 | 70%可靠性参考时点 | 90%可靠性参考时点 | Minimax最佳时点 | 平均可靠性 | 10%分位可靠性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

## 组数冻结与 Bootstrap 证据

- patient-level bootstrap OOB 审计中，K=3 落在平均 OOB error 最低的 K=4 的 1-SE 范围内；`min_group_size=20/25` 均按 1-SE rule 选择更简洁的 K=3。
- 低/中与高 BMI 分界落在各自全样本分界 ±1 BMI 内的比例分别为 67.8% 与 98.6%；三组 Minimax 时点单调不减比例在主审计/复核中为 100.0%/99.8%。

## 多因素作用

- Height：EDF={height_row['edf']:.4f}，p={height_row[p_column]:.4g}，k-index={height_k[k_column]:.4f}；平滑效应为{height_direction}。
{chr(10).join(clinical_lines)}

## Q2 → Q3

- 最终 K={result.k}；Q2→Q3：{boundary_text}；按 BMI 区间映射的最大 Minimax 时点移动为 {max_q2_q3_timing:.1f} 周。
- 完整 K、分界、人数与 50%/70%/90%/Minimax 对照见 `q2_vs_q3_comparison.xlsx`。

## 个体化损失

- mean |Δ|={loss['mean_abs_delta']:.3f} 周，median |Δ|={loss['median_abs_delta']:.3f} 周，90% quantile |Δ|={loss['q90_abs_delta']:.3f} 周，max |Δ|={loss['max_abs_delta']:.3f} 周。
- |Δ|≤0.5/1.0/2.0 周的比例分别为 {loss['proportion_abs_delta_le_0_5']:.1%}、{loss['proportion_abs_delta_le_1_0']:.1%}、{loss['proportion_abs_delta_le_2_0']:.1%}。

## 测量误差与安全性

- 复用 Q2 四个误差水平后，Minimax 时点最多移动 {max_time_shift:.1f} 周，候选分界最多移动 {max_cut_shift:.4f} BMI 单位；主模型和无误差主分组未改变。
- 组内尾部最低的是 G{int(tail['Group'])}：Minimax 时点下 10% 分位可靠性={tail['q10_reliability']:.4f}，最小可靠性={tail['min_reliability']:.4f}。未据此临时修改分组规则。

`t50/t70/t90` 仅为可靠性参考时点；最终推荐为 Minimax 时点。
"""
    (DECISION_DIR / "q3_summary.md").write_text(summary, encoding="utf-8")


def self_check() -> None:
    assert math.isclose(parse_ga("11w+6"), 11 + 6 / 7)
    assert np.allclose(delay_risk(np.array([10, 12, 13, 27])), [0, 0, 1 / 15, 1])
    test = np.array([0.2, 0.5, 0.8])
    assert int(np.argmin(np.maximum(1 - test, np.array([0.0, 0.2, 0.8])))) == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-temp", type=Path, required=True)
    args = parser.parse_args()
    self_check()
    np.random.seed(SEED)
    for directory in [DATA_DIR, RAW_DIR, DECISION_DIR, FIGURE_DIR, BOUNDARY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    frozen_before = {name: tree_hash(WORK_PLACE / name) for name in ["Q1", "Q2"]}
    print("[1/6] augment frozen Q1 data with Height", flush=True)
    _, height_audit = prepare_q3_data()
    try:
        print("[2/6] fit fixed Q3 Beta-GAMM and export patient eta", flush=True)
        r_runtime = run_r_model(args.r_temp.resolve())
        info = read_model_info()
        profiles = pd.read_csv(BOUNDARY_DIR / "patient_profiles.csv").sort_values("patient_id").reset_index(drop=True)
        patient_eta = pd.read_csv(BOUNDARY_DIR / "patient_eta_surface.csv")
        smooth = pd.read_csv(BOUNDARY_DIR / "smooth_terms.csv")
        parametric = pd.read_csv(BOUNDARY_DIR / "parametric_terms.csv")
        k_check = pd.read_csv(BOUNDARY_DIR / "k_check.csv")
        height_effect = pd.read_csv(BOUNDARY_DIR / "height_effect.csv")
        if len(profiles) != 267 or len(patient_eta) != 267 * 151:
            raise AssertionError("R-to-Python patient surface contract failed")

        print("[3/6] integrate individual reliability and rerun frozen DP", flush=True)
        error_scenarios = pd.read_csv(Q2_ROOT / "data_processed" / "error_scenarios.csv")
        measurement_sds = dict(zip(error_scenarios["scenario"], error_scenarios["measurement_sd"], strict=True))
        phi = float(info["precision"])
        sigma_u2 = float(info["random_intercept_variance"])
        nodes, _ = hermgauss(30)
        u_nodes = np.sqrt(2 * sigma_u2) * nodes
        eta_values = patient_eta["eta_base"].to_numpy(float)
        engine = ProbabilityEngine.create(
            phi=phi,
            sigma_u2=sigma_u2,
            empirical_offsets=np.array([0.0]),
            eta_min=float(eta_values.min() + u_nodes.min()),
            eta_max=float(eta_values.max() + u_nodes.max()),
            measurement_sds=measurement_sds,
        )
        for scenario in measurement_sds:
            patient_eta[f"p_{scenario}"] = engine.evaluate_eta(
                eta_values, scenario=scenario, covariates={"z_offsets": np.array([0.0])}
            )
        ga_grid = np.sort(patient_eta["GA"].unique())
        patient_ids = profiles["patient_id"].to_numpy()
        curve_matrices = {
            scenario: patient_eta.pivot(index="patient_id", columns="GA", values=f"p_{scenario}")
            .reindex(index=patient_ids, columns=ga_grid)
            .to_numpy()
            for scenario in measurement_sds
        }
        if any(matrix.shape != (267, 151) or np.isnan(matrix).any() for matrix in curve_matrices.values()):
            raise AssertionError("individual reliability curve matrix contract failed")
        segmentations = {
            scenario: optimal_segmentations(
                profiles["BMI"].to_numpy(float),
                matrix,
                ga_grid,
                k_values=(FINAL_K,),
                min_group_size=MIN_GROUP_SIZE,
            )
            for scenario, matrix in curve_matrices.items()
        }
        final_k = FINAL_K
        final_result = segmentations["no_error"][final_k]
        summary_parts, segment_parts = [], []
        for scenario, results in segmentations.items():
            summary, segments = segmentation_frames(results, scenario, final_k if scenario == "no_error" else -1)
            summary_parts.append(summary)
            segment_parts.append(segments)
        segmentation_summary = pd.concat(summary_parts, ignore_index=True)
        segments_all = pd.concat(segment_parts, ignore_index=True)

        print("[4/6] compute Minimax, individual loss, Q2 comparison and sensitivity", flush=True)
        final_decisions, final_risks = group_decisions(
            profiles, curve_matrices["no_error"], ga_grid, final_result, "no_error"
        )
        individual, loss_summary = individual_timing(
            profiles, curve_matrices["no_error"], ga_grid, final_result, final_decisions
        )
        overview, comparison_long, comparison_shifts = q2_q3_comparison(profiles, final_decisions, final_result)
        sensitivity_parts, error_segment_rows = [], []
        baseline_times = final_decisions.set_index("Group")["minimax_GA"]
        baseline_cuts = np.asarray(final_result.cutpoints)
        for scenario, matrix in curve_matrices.items():
            result = segmentations[scenario][final_k]
            decision, _ = group_decisions(profiles, matrix, ga_grid, result, scenario)
            cuts = np.asarray(result.cutpoints)
            decision["measurement_sd"] = measurement_sds[scenario]
            decision["cutpoints"] = "; ".join(f"{value:.4f}" for value in cuts)
            decision["max_cutpoint_shift_vs_no_error"] = float(np.max(np.abs(cuts - baseline_cuts)))
            decision["timing_shift_vs_no_error"] = decision.apply(
                lambda row: row["minimax_GA"] - baseline_times.loc[row["Group"]], axis=1
            )
            sensitivity_parts.append(decision)
            for group_id, segment in enumerate(result.segments, 1):
                error_segment_rows.append(
                    {
                        "scenario": scenario,
                        "measurement_sd": measurement_sds[scenario],
                        "K": final_k,
                        "Group": group_id,
                        "BMI_interval": interval_label(result, group_id),
                        "N": segment.n_patients,
                        "cutpoints": "; ".join(f"{value:.4f}" for value in cuts),
                        "group_curve_cost": segment.cost,
                    }
                )
        sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
        error_segments = pd.DataFrame(error_segment_rows)

        groups = assign_groups(profiles["BMI"].to_numpy(float), final_result)
        individual_curves = patient_eta[["patient_id", "GA", "p_no_error"]].rename(columns={"p_no_error": "reliability"})
        individual_curves = individual_curves.merge(profiles, on="patient_id", validate="many_to_one")
        individual_curves["Group"] = individual_curves["patient_id"].map(dict(zip(patient_ids, groups, strict=True)))
        measurement_curves = pd.concat(
            [
                patient_eta[["patient_id", "GA", f"p_{scenario}"]]
                .rename(columns={f"p_{scenario}": "reliability"})
                .assign(scenario=scenario, measurement_sd=measurement_sds[scenario])
                .merge(profiles[["patient_id", "BMI", "Height"]], on="patient_id", validate="many_to_one")
                for scenario in measurement_sds
            ],
            ignore_index=True,
        )
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
                "t50": "50%时点",
                "t70": "70%时点",
                "t90": "90%时点",
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
                "组别", "BMI区间", "人数", "50%时点", "70%时点", "90%时点", "Minimax最佳时点", "周+天",
                "平均可靠性", "10%分位可靠性", "最小可靠性", "检测失败风险", "延迟风险", "Minimax风险",
            ]
        ]

        print("[5/6] save auditable data, four figures and summary", flush=True)
        frames = {
            "patient_profiles": profiles,
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
            "model_smooth_terms": smooth,
            "model_parametric_terms": parametric,
            "model_k_check": k_check,
            "height_effect": height_effect,
        }
        for name, frame in frames.items():
            frame.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
        save_figures(
            profiles, curve_matrices["no_error"], ga_grid, final_result, final_decisions, final_risks, comparison_long
        )
        write_summary(
            final_decisions,
            final_result,
            overview,
            comparison_shifts,
            loss_summary,
            sensitivity,
            smooth,
            k_check,
            parametric,
            height_effect,
        )

        frozen_after = {name: tree_hash(WORK_PLACE / name) for name in ["Q1", "Q2"]}
        if frozen_after != frozen_before:
            raise AssertionError("Q1 or Q2 changed during Q3")
        manifest = {
            "seed": SEED,
            "frozen_contracts_before": frozen_before,
            "frozen_contracts_after": frozen_after,
            "source_xlsx_sha256": sha256(SOURCE_XLSX),
            "q1_data_sha256": sha256(Q1_DATA),
            "height_audit": height_audit,
            "model_formula": info["formula"],
            "n_patients": 267,
            "ga_grid": {"min": 10.0, "max": 25.0, "step": 0.1},
            "candidate_K": [FINAL_K],
            "selected_K": final_k,
            "min_group_size": MIN_GROUP_SIZE,
            "K_selection_basis": "patient-level bootstrap OOB + 1-SE rule",
            "random_effect_handling": "30-node Gauss-Hermite marginalization with zero fixed-covariate offset",
            "measurement_error_source": "Q2/data_processed/error_scenarios.csv",
            "r_runtime_stderr": r_runtime["stderr"],
        }
        (DATA_DIR / "q3_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_frame = pd.DataFrame(
            [
                {"key": "Q1_tree_sha256", "value": frozen_before["Q1"]["sha256"]},
                {"key": "Q2_tree_sha256", "value": frozen_before["Q2"]["sha256"]},
                {"key": "source_xlsx_sha256", "value": manifest["source_xlsx_sha256"]},
                {"key": "selected_K", "value": final_k},
                {"key": "seed", "value": SEED},
            ]
        )
        model_info = pd.DataFrame({"key": list(info), "value": list(info.values())})
        compact = [
            ("model_info", model_info),
            ("smooth_terms", smooth),
            ("parametric_terms", parametric),
            ("k_check", k_check),
            ("K_summary", segmentation_summary),
            ("final_timing", final_decisions),
            ("individual_loss", loss_summary),
            ("Q2_Q3_overview", overview),
            ("sensitivity", sensitivity),
            ("manifest", manifest_frame),
        ]
        specifications = [
            {
                "path": "raw/individual_reliability_curves.xlsx",
                "sheets": [("individual_curves", individual_curves), ("patient_profiles", profiles)],
            },
            {
                "path": "raw/q3_segmentation_all_K.xlsx",
                "sheets": [("K_summary", segmentation_summary), ("segments_all_K", segments_all)],
            },
            {
                "path": "raw/individual_optimal_timing.xlsx",
                "sheets": [("individual_timing", individual), ("loss_summary", loss_summary)],
            },
            {
                "path": "raw/q3_measurement_error_raw.xlsx",
                "sheets": [
                    ("error_scenarios", error_scenarios),
                    *[
                        (f"curves_{scenario.replace('measurement_', 'm')}", measurement_curves[measurement_curves["scenario"] == scenario])
                        for scenario in measurement_sds
                    ],
                    ("error_segments", error_segments),
                    ("error_timing", sensitivity),
                ],
            },
            {"path": "raw/q3_all_raw.xlsx", "sheets": compact},
            {"path": "decision/final_q3_bmi_groups.xlsx", "sheets": [("final_groups", final_groups)]},
            {"path": "decision/final_q3_nipt_timing.xlsx", "sheets": [("final_timing", final_nipt)]},
            {
                "path": "decision/q2_vs_q3_comparison.xlsx",
                "sheets": [("overview", overview), ("group_comparison", comparison_long), ("shifts", comparison_shifts)],
            },
            {"path": "decision/individualization_loss.xlsx", "sheets": [("loss_summary", loss_summary), ("individual_timing", individual)]},
            {"path": "decision/q3_sensitivity_summary.xlsx", "sheets": [("sensitivity", sensitivity), ("error_segments", error_segments)]},
        ]
        build_payload(
            [
                {
                    "path": specification["path"],
                    "sheets": [{"name": name, **json_ready(frame)} for name, frame in specification["sheets"]],
                }
                for specification in specifications
            ]
        )
        print(f"[6/6] Q3 analysis complete: K={final_k}, payload={PAYLOAD_PATH}", flush=True)
    finally:
        pass


if __name__ == "__main__":
    main()
