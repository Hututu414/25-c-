from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.polynomial.hermite import hermgauss
from optimize_bmi_groups import (
    ProbabilityEngine,
    assign_groups,
    choose_k,
    optimal_segmentations,
    probability_kernel,
    self_check,
    timing_audit,
)

Q2_ROOT = Path(__file__).resolve().parents[1]
WORK_PLACE = Q2_ROOT.parent
PROJECT_ROOT = WORK_PLACE.parent
Q1_ROOT = WORK_PLACE / "Q1"
Q1_DATA = Q1_ROOT / "data_processed" / "q1_round2_model_data.csv"
Q1_MODEL = Q1_ROOT / "code" / "m4_round2.R"
SOURCE_XLSX = PROJECT_ROOT / "C题" / "附件.xlsx"
R_ADAPTER = Q2_ROOT / "code" / "reliability_from_m4.R"
DATA_DIR = Q2_ROOT / "data_processed"
OUTPUT_DIR = Q2_ROOT / "outputs"
RAW_DIR = OUTPUT_DIR / "raw"
DECISION_DIR = OUTPUT_DIR / "decision"
FIGURE_DIR = OUTPUT_DIR / "figures"
PAYLOAD_PATH = OUTPUT_DIR / ".artifact_payload.json"

Q_LEVELS = (0.90, 0.95, 0.975)
MIN_GROUP_SIZE = 30
SEED = 20260824
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_reset(directory: Path) -> None:
    resolved = directory.resolve()
    if Q2_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"refusing to reset outside Q2: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def find_rscript() -> Path:
    found = shutil.which("Rscript")
    if found:
        return Path(found)
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = sorted((program_files / "R").glob("R-*/bin/Rscript.exe"), reverse=True)
    if not candidates:
        raise RuntimeError("Rscript not found")
    return candidates[0]


def decode_output(data: bytes) -> str:
    for encoding in ("utf-8", "gb18030", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def run_r_adapter(r_temp: Path) -> tuple[Path, dict[str, str]]:
    if not r_temp.is_absolute() or any(ord(character) > 127 for character in str(r_temp)):
        raise ValueError("--r-temp must be an absolute writable ASCII path")
    r_temp.mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix="q2_m4_", dir=r_temp))
    output = runtime / "output"
    shutil.copy2(R_ADAPTER, runtime / R_ADAPTER.name)
    shutil.copy2(Q1_DATA, runtime / "model_data.csv")
    env = os.environ.copy()
    env.update(
        {"TEMP": str(r_temp), "TMP": str(r_temp), "TMPDIR": str(r_temp), "LC_ALL": "C", "LANG": "C"}
    )
    process = subprocess.run(
        [
            str(find_rscript()),
            "--vanilla",
            str(runtime / R_ADAPTER.name),
            str(runtime / "model_data.csv"),
            str(output),
        ],
        cwd=runtime,
        env=env,
        capture_output=True,
        shell=False,
        check=False,
    )
    error_path = output / "error.txt"
    error = error_path.read_text(encoding="utf-8", errors="replace") if error_path.exists() else ""
    if process.returncode:
        raise RuntimeError(error or decode_output(process.stderr))
    boundary = DATA_DIR / "m4_boundary"
    safe_reset(boundary)
    shutil.copytree(output, boundary, dirs_exist_ok=True)
    return runtime, {"stderr": decode_output(process.stderr).strip()}


def read_info(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, keep_default_na=False)
    return dict(zip(frame["key"].astype(str), frame["value"].astype(str), strict=True))


def estimate_measurement_error() -> tuple[pd.DataFrame, float, float]:
    raw = pd.read_excel(SOURCE_XLSX, sheet_name="男胎检测数据", engine="openpyxl")
    raw.columns = [str(column).strip() for column in raw.columns]
    required = ["孕妇代码", "检测抽血次数", "检测孕周", "Y染色体浓度"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"source workbook misses repeat columns: {missing}")
    patient_values = sorted(raw["孕妇代码"].astype(str).unique())
    patient_map = {value: f"P{index:04d}" for index, value in enumerate(patient_values, 1)}
    raw["patient_id"] = raw["孕妇代码"].astype(str).map(patient_map)
    raw["Y"] = pd.to_numeric(raw["Y染色体浓度"], errors="raise")
    repeat = (
        raw.groupby(["patient_id", "检测抽血次数", "检测孕周"], as_index=False)["Y"]
        .agg(n="count", mean_y="mean", within_sd="std", min_y="min", max_y="max")
        .query("n > 1")
        .rename(columns={"检测抽血次数": "blood_draw_no", "检测孕周": "GA_raw"})
    )
    repeat["within_variance"] = repeat["within_sd"] ** 2
    degrees_freedom = int((repeat["n"] - 1).sum())
    pooled_variance = float(((repeat["n"] - 1) * repeat["within_variance"]).sum() / degrees_freedom)
    pooled_sd = float(np.sqrt(pooled_variance))
    if len(repeat) != 39 or degrees_freedom != 60:
        raise AssertionError("technical-repeat contract changed")
    return repeat, pooled_variance, pooled_sd


def segmentation_frames(
    results: dict[int, object],
    scenario: str,
    selected_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, segment_rows = [], []
    previous_objective = np.nan
    for k in sorted(results):
        result = results[k]
        improvement = (
            (previous_objective - result.objective) / previous_objective
            if np.isfinite(previous_objective)
            else np.nan
        )
        summary_rows.append(
            {
                "scenario": scenario,
                "K": k,
                "total_curve_error": result.objective,
                "marginal_improvement_pct": 100 * improvement,
                "min_group_N": min(segment.n_patients for segment in result.segments),
                "max_group_N": max(segment.n_patients for segment in result.segments),
                "cutpoints": "; ".join(f"{value:.4f}" for value in result.cutpoints),
                "selected": k == selected_k,
            }
        )
        for group_id, segment in enumerate(result.segments, 1):
            lower = segment.bmi_min if group_id == 1 else result.cutpoints[group_id - 2]
            upper = segment.bmi_max if group_id == k else result.cutpoints[group_id - 1]
            segment_rows.append(
                {
                    "scenario": scenario,
                    "K": k,
                    "Group": group_id,
                    "BMI_lower": lower,
                    "BMI_upper": upper,
                    "observed_BMI_min": segment.bmi_min,
                    "observed_BMI_max": segment.bmi_max,
                    "N": segment.n_patients,
                    "group_curve_cost": segment.cost,
                }
            )
        previous_objective = result.objective
    return pd.DataFrame(summary_rows), pd.DataFrame(segment_rows)


def direct_mixture(
    base_eta: np.ndarray,
    eta_grid: np.ndarray,
    kernel: np.ndarray,
    offsets: np.ndarray,
    sigma_u2: float,
    nodes_n: int,
) -> np.ndarray:
    unique_offsets, counts = np.unique(np.round(offsets, 12), return_counts=True)
    nodes, weights = hermgauss(nodes_n)
    u_nodes = np.sqrt(2.0 * sigma_u2) * nodes
    u_weights = weights / np.sqrt(np.pi)
    shifts = (unique_offsets[:, None] + u_nodes[None, :]).ravel()
    mix_weights = ((counts / counts.sum())[:, None] * u_weights[None, :]).ravel()
    eta = base_eta[:, None] + shifts[None, :]
    return np.interp(eta.ravel(), eta_grid, kernel).reshape(eta.shape) @ mix_weights


def configure_figures() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save_figures(
    plot_surface: pd.DataFrame,
    profiles: pd.DataFrame,
    baseline_curves: np.ndarray,
    ga_grid: np.ndarray,
    final_result: object,
    final_timing: pd.DataFrame,
    timing_all_q: pd.DataFrame,
    error_timing: pd.DataFrame,
) -> None:
    configure_figures()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    surface = plot_surface.pivot(index="BMI", columns="GA", values="p_no_error").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    contour = ax.contourf(
        surface.columns,
        surface.index,
        surface.to_numpy(),
        levels=np.linspace(0, 1, 21),
        cmap="cividis",
        vmin=0,
        vmax=1,
    )
    if surface.to_numpy().min() <= 0.95 <= surface.to_numpy().max():
        line = ax.contour(surface.columns, surface.index, surface.to_numpy(), levels=[0.95], colors="white", linewidths=1.4)
        ax.clabel(line, fmt={0.95: "95%"}, fontsize=8)
    fig.colorbar(contour, ax=ax, label=r"$P(Y\geq 4\%)$")
    ax.set(xlabel="Gestational age (weeks)", ylabel=r"Baseline BMI (kg/m$^2$)", title="Population-standardized reliability surface")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "reliability_surface.png", bbox_inches="tight")
    plt.close(fig)

    groups = assign_groups(profiles["BMI"].to_numpy(float), final_result)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for group_id, row in final_timing.set_index("Group").iterrows():
        selected = groups == group_id
        mean_curve = baseline_curves[selected].mean(axis=0)
        ax.plot(
            ga_grid,
            mean_curve,
            color=OKABE_ITO[(group_id - 1) % len(OKABE_ITO)],
            linewidth=2,
            label=f"G{group_id}: {row['BMI_interval']} (n={int(row['N'])})",
        )
        if np.isfinite(row["recommended_GA"]):
            ax.scatter(row["recommended_GA"], row["mean_reliability"], s=35, color=OKABE_ITO[(group_id - 1) % len(OKABE_ITO)], zorder=3)
    ax.axhline(0.95, color="#333333", linestyle="--", linewidth=1, label="95% reliability")
    ax.set(xlim=(10, 25), ylim=(0, 1.01), xlabel="Gestational age (weeks)", ylabel=r"$P(Y\geq 4\%)$", title="Final BMI-group reliability curves")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bmi_reliability_curves_and_groups.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    axes[0].scatter(final_timing["recommended_GA"], final_timing["Group"], s=55, color=OKABE_ITO[0])
    for _, row in final_timing.iterrows():
        axes[0].text(row["recommended_GA"] + 0.12, row["Group"], f"{row['recommended_GA']:.1f} w", va="center", fontsize=8)
    axes[0].set(xlabel="Earliest recommended GA (weeks)", ylabel="BMI group", yticks=final_timing["Group"], title="Recommended timing")
    for metric, marker, color in [
        ("mean_reliability", "o", OKABE_ITO[0]),
        ("q10_reliability", "s", OKABE_ITO[1]),
        ("min_reliability", "^", OKABE_ITO[2]),
    ]:
        axes[1].scatter(final_timing[metric], final_timing["Group"], marker=marker, s=42, color=color, label=metric.replace("_reliability", ""))
    axes[1].axvline(0.95, color="#333333", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Reliability at recommended GA", ylabel="", yticks=final_timing["Group"], title="Within-group safety audit")
    axes[1].legend(frameon=False)
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "final_group_timing.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)
    for group_id, frame in timing_all_q.groupby("Group"):
        axes[0].plot(frame["q"], frame["recommended_GA"], marker="o", color=OKABE_ITO[(group_id - 1) % len(OKABE_ITO)], label=f"G{group_id}")
    axes[0].set(xlabel="Required reliability q", ylabel="Recommended GA (weeks)", title="Reliability-level sensitivity")
    scenario_order = ["no_error", "measurement_0_75", "measurement_1_00", "measurement_1_25"]
    labels = ["0", "0.75σ", "1.00σ", "1.25σ"]
    for group_id, frame in error_timing.groupby("Group"):
        indexed = frame.set_index("scenario").reindex(scenario_order)
        axes[1].plot(range(len(labels)), indexed["recommended_GA"], marker="s", color=OKABE_ITO[(group_id - 1) % len(OKABE_ITO)], label=f"G{group_id}")
    axes[1].set(xlabel="Measurement-error scale", xticks=range(len(labels)), xticklabels=labels, title="Measurement-error sensitivity")
    axes[0].legend(frameon=False, ncol=2)
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sensitivity_analysis.png", bbox_inches="tight")
    plt.close(fig)


def json_ready(frame: pd.DataFrame) -> dict[str, object]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return {"columns": list(clean.columns), "rows": clean.values.tolist()}


def build_payload(specs: list[dict[str, object]]) -> None:
    PAYLOAD_PATH.write_text(json.dumps({"workbooks": specs}, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-temp", type=Path, required=True)
    args = parser.parse_args()
    np.random.seed(SEED)
    self_check()
    for directory in (DATA_DIR, RAW_DIR, DECISION_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    input_hashes_before = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in (Q1_DATA, Q1_MODEL, SOURCE_XLSX)}
    print("[1/5] refit frozen Q1 M4", flush=True)
    runtime, r_runtime = run_r_adapter(args.r_temp.resolve())
    try:
        boundary = DATA_DIR / "m4_boundary"
        info = read_info(boundary / "model_info.csv")
        if int(float(info["n_obs"])) != 1022 or int(float(info["n_patients"])) != 267:
            raise AssertionError("R boundary row or patient count changed")
        conditional_r2 = float(info["conditional_r2"])
        if abs(conditional_r2 - 0.8263662212036017) > 1e-5:
            raise AssertionError("Q1 conditional R2 sanity check changed")
        profiles = pd.read_csv(boundary / "patient_profiles.csv").sort_values("patient_id").reset_index(drop=True)
        patient_eta = pd.read_csv(boundary / "patient_eta_surface.csv")
        plot_eta = pd.read_csv(boundary / "plot_eta_surface.csv")
        if len(profiles) != 267 or patient_eta.groupby("patient_id").size().nunique() != 1:
            raise AssertionError("invalid R-to-Python profile/surface contract")

        print("[2/5] estimate repeat-measurement error and integrate reliability", flush=True)
        repeat_groups, pooled_variance, pooled_sd = estimate_measurement_error()
        measurement_sds = {
            "no_error": 0.0,
            "measurement_0_75": 0.75 * pooled_sd,
            "measurement_1_00": pooled_sd,
            "measurement_1_25": 1.25 * pooled_sd,
        }
        phi = float(info["precision"])
        sigma_u2 = float(info["random_intercept_variance"])
        nodes, _ = hermgauss(30)
        mix_min = profiles["z_offset"].min() + np.sqrt(2 * sigma_u2) * nodes.min()
        mix_max = profiles["z_offset"].max() + np.sqrt(2 * sigma_u2) * nodes.max()
        all_base = np.r_[patient_eta["eta_base"].to_numpy(), plot_eta["eta_base"].to_numpy()]
        engine = ProbabilityEngine.create(
            phi=phi,
            sigma_u2=sigma_u2,
            empirical_offsets=profiles["z_offset"].to_numpy(),
            eta_min=float(all_base.min() + mix_min),
            eta_max=float(all_base.max() + mix_max),
            measurement_sds=measurement_sds,
        )

        kernel_rows = []
        test_eta = np.linspace(engine.eta_grid[0], engine.eta_grid[-1], 101)
        for scenario, sd in measurement_sds.items():
            interpolated = np.interp(test_eta, engine.eta_grid, engine.kernels[scenario])
            exact = probability_kernel(test_eta, phi, sd, error_nodes=21)
            error31 = probability_kernel(test_eta, phi, sd, error_nodes=31)
            kernel_rows.append(
                {
                    "scenario": scenario,
                    "measurement_sd": sd,
                    "max_interpolation_error": float(np.max(np.abs(interpolated - exact))),
                    "max_21_vs_31_error_quadrature": float(np.max(np.abs(exact - error31))),
                }
            )
        base_test = np.quantile(all_base, np.linspace(0.05, 0.95, 25))
        for scenario in measurement_sds:
            p30 = direct_mixture(base_test, engine.eta_grid, engine.kernels[scenario], profiles["z_offset"].to_numpy(), sigma_u2, 30)
            p40 = direct_mixture(base_test, engine.eta_grid, engine.kernels[scenario], profiles["z_offset"].to_numpy(), sigma_u2, 40)
            for row in kernel_rows:
                if row["scenario"] == scenario:
                    row["max_30_vs_40_random_effect_quadrature"] = float(np.max(np.abs(p30 - p40)))
        kernel_validation = pd.DataFrame(kernel_rows)
        if kernel_validation[["max_interpolation_error", "max_21_vs_31_error_quadrature", "max_30_vs_40_random_effect_quadrature"]].max().max() > 2e-4:
            raise AssertionError("probability integration convergence check failed")

        for scenario in measurement_sds:
            patient_eta[f"p_{scenario}"] = engine.evaluate_eta(patient_eta["eta_base"].to_numpy(), scenario)
            plot_eta[f"p_{scenario}"] = engine.evaluate_eta(plot_eta["eta_base"].to_numpy(), scenario)

        ga_grid = np.sort(patient_eta["GA"].unique())
        patient_ids = profiles["patient_id"].to_numpy()
        curve_matrices = {}
        for scenario in measurement_sds:
            curve_matrices[scenario] = (
                patient_eta.pivot(index="patient_id", columns="GA", values=f"p_{scenario}")
                .reindex(index=patient_ids, columns=ga_grid)
                .to_numpy()
            )
        if any(matrix.shape != (267, 151) or np.isnan(matrix).any() for matrix in curve_matrices.values()):
            raise AssertionError("reliability curve matrix contract failed")

        print("[3/5] globally optimize contiguous BMI groups", flush=True)
        all_segmentations = {}
        segmentation_summary_parts, segment_parts = [], []
        for scenario, curves in curve_matrices.items():
            results = optimal_segmentations(
                profiles["BMI"].to_numpy(), curves, ga_grid, min_group_size=MIN_GROUP_SIZE
            )
            all_segmentations[scenario] = results
            selected = choose_k(results) if scenario == "no_error" else -1
            summary, segments = segmentation_frames(results, scenario, selected)
            segmentation_summary_parts.append(summary)
            segment_parts.append(segments)
        segmentation_summary = pd.concat(segmentation_summary_parts, ignore_index=True)
        segments_all = pd.concat(segment_parts, ignore_index=True)
        final_k = choose_k(all_segmentations["no_error"])
        final_result = all_segmentations["no_error"][final_k]

        timing_parts = [
            timing_audit(
                profiles,
                curve_matrices["no_error"],
                ga_grid,
                final_result,
                q,
                "no_error",
            )
            for q in Q_LEVELS
        ]
        timing_all_q = pd.concat(timing_parts, ignore_index=True)
        final_timing = timing_all_q[np.isclose(timing_all_q["q"], 0.95)].copy()

        error_timing_parts, error_segment_rows = [], []
        baseline_cutpoints = np.asarray(final_result.cutpoints)
        baseline_timing = final_timing.set_index("Group")["recommended_GA"]
        for scenario, curves in curve_matrices.items():
            result = all_segmentations[scenario][final_k]
            timing = timing_audit(profiles, curves, ga_grid, result, 0.95, scenario)
            cutpoints = np.asarray(result.cutpoints)
            timing["cutpoints"] = "; ".join(f"{value:.4f}" for value in cutpoints)
            timing["max_cutpoint_shift_vs_no_error"] = float(np.max(np.abs(cutpoints - baseline_cutpoints)))
            timing["timing_shift_vs_no_error"] = timing.apply(
                lambda row: row["recommended_GA"] - baseline_timing.loc[row["Group"]], axis=1
            )
            error_timing_parts.append(timing)
            for group_id, segment in enumerate(result.segments, 1):
                error_segment_rows.append(
                    {
                        "scenario": scenario,
                        "measurement_sd": measurement_sds[scenario],
                        "K": final_k,
                        "Group": group_id,
                        "N": segment.n_patients,
                        "observed_BMI_min": segment.bmi_min,
                        "observed_BMI_max": segment.bmi_max,
                        "group_curve_cost": segment.cost,
                        "cutpoints": "; ".join(f"{value:.4f}" for value in result.cutpoints),
                    }
                )
        error_timing = pd.concat(error_timing_parts, ignore_index=True)
        error_segments = pd.DataFrame(error_segment_rows)

        sensitivity_q = timing_all_q.copy()
        sensitivity_q["analysis_type"] = "reliability_level"
        sensitivity_q["measurement_sd"] = 0.0
        sensitivity_q["cutpoints"] = "; ".join(f"{value:.4f}" for value in final_result.cutpoints)
        sensitivity_q["timing_shift_vs_q95_no_error"] = sensitivity_q.apply(
            lambda row: row["recommended_GA"] - baseline_timing.loc[row["Group"]], axis=1
        )
        sensitivity_error = error_timing.copy()
        sensitivity_error["analysis_type"] = "measurement_error"
        sensitivity_error["measurement_sd"] = sensitivity_error["scenario"].map(measurement_sds)
        sensitivity_error["timing_shift_vs_q95_no_error"] = sensitivity_error["timing_shift_vs_no_error"]
        sensitivity_summary = pd.concat([sensitivity_q, sensitivity_error], ignore_index=True, sort=False)

        print("[4/5] write auditable data and publication figures", flush=True)
        population_curves = patient_eta[["patient_id", "BMI", "GA", "p_no_error"]].rename(columns={"p_no_error": "reliability"})
        surface_grid = plot_eta[["BMI", "GA", "p_no_error"]].rename(columns={"p_no_error": "reliability"})
        measurement_curves = pd.concat(
            [
                patient_eta[["patient_id", "BMI", "GA", f"p_{scenario}"]]
                .rename(columns={f"p_{scenario}": "reliability"})
                .assign(scenario=scenario, measurement_sd=measurement_sds[scenario])
                for scenario in measurement_sds
            ],
            ignore_index=True,
        )
        error_scenarios = pd.DataFrame(
            [{"scenario": scenario, "multiplier": 0.0 if pooled_sd == 0 else sd / pooled_sd, "measurement_sd": sd} for scenario, sd in measurement_sds.items()]
        )
        model_info = pd.DataFrame({"key": list(info), "value": list(info.values())})

        final_groups = final_timing[
            ["Group", "BMI_interval", "N", "group_curve_cost", "recommended_GA", "week_day", "mean_reliability"]
        ].rename(
            columns={
                "Group": "组别",
                "BMI_interval": "BMI区间",
                "N": "孕妇数",
                "group_curve_cost": "组内曲线误差",
                "recommended_GA": "推荐时点",
                "week_day": "周+天",
                "mean_reliability": "平均可靠性",
            }
        )
        final_nipt = final_timing[
            ["Group", "BMI_interval", "recommended_GA", "week_day", "mean_reliability", "q10_reliability", "min_reliability", "proportion_at_or_above_q"]
        ].rename(
            columns={
                "Group": "组别",
                "BMI_interval": "BMI区间",
                "recommended_GA": "最佳孕周",
                "week_day": "周+天",
                "mean_reliability": "平均可靠性",
                "q10_reliability": "10%分位可靠性",
                "min_reliability": "最小可靠性",
                "proportion_at_or_above_q": "达到95%的BMI比例",
            }
        )

        frames = {
            "population_curves": population_curves,
            "surface_grid": surface_grid,
            "patient_profiles": profiles,
            "model_info": model_info,
            "kernel_validation": kernel_validation,
            "segmentation_summary": segmentation_summary,
            "segments_all": segments_all,
            "timing_all_q": timing_all_q,
            "repeat_groups": repeat_groups,
            "error_scenarios": error_scenarios,
            "measurement_curves": measurement_curves,
            "error_segments": error_segments,
            "error_timing": error_timing,
            "sensitivity_summary": sensitivity_summary,
            "final_groups": final_groups,
            "final_nipt": final_nipt,
        }
        for name, frame in frames.items():
            frame.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

        save_figures(plot_eta, profiles, curve_matrices["no_error"], ga_grid, final_result, final_timing, timing_all_q, error_timing)

        q95_rows = final_timing.sort_values("Group")
        q90 = timing_all_q[np.isclose(timing_all_q["q"], 0.90)].set_index("Group")["recommended_GA"]
        q975 = timing_all_q[np.isclose(timing_all_q["q"], 0.975)].set_index("Group")["recommended_GA"]
        max_q_shift = float(np.nanmax(q975 - q90))
        error_only = error_timing[error_timing["scenario"] != "no_error"]
        max_error_timing_shift = float(np.nanmax(np.abs(error_only["timing_shift_vs_no_error"])))
        max_error_cut_shift = float(error_only["max_cutpoint_shift_vs_no_error"].max())
        any_below = bool(q95_rows["any_below_q"].any())
        marked_tail = bool(q95_rows["marked_tail_failure"].any())
        selected_row = segmentation_summary[(segmentation_summary["scenario"] == "no_error") & (segmentation_summary["K"] == final_k)].iloc[0]
        previous = segmentation_summary[(segmentation_summary["scenario"] == "no_error") & (segmentation_summary["K"] == final_k - 1)]
        improvement_text = "采用曲线误差肘部解"
        if len(previous):
            improvement_text = (
                f"相对 K={final_k - 1}，曲线误差下降 "
                f"{selected_row['marginal_improvement_pct']:.2f}%"
            )
        group_lines = "\n".join(
            f"| G{int(row.Group)} | {row.BMI_interval} | {int(row.N)} | {row.recommended_GA:.1f} ({row.week_day}) | {row.mean_reliability:.4f} |"
            for row in q95_rows.itertuples()
        )
        summary_md = f"""# Q2 决策摘要

## 最终 BMI 分组

| Group | BMI interval | N | recommended GA | reliability |
|---|---:|---:|---:|---:|
{group_lines}

## 模型选择

- 最终 K = **{final_k}**。
- 选择依据：连续分段 DP 的对数误差肘部；{improvement_text}；最小组样本量为 {int(selected_row['min_group_N'])}，并同时检查测量误差下分界稳定性。未因目标函数单调下降而机械选择 K=5。
- Q1 sanity check：N=1022、孕妇数=267、conditional in-sample R²={conditional_r2:.6f}。

## 时点

上表给出 q=95% 时各组满足群体平均可靠性约束的最早 NIPT 时点；计算网格为 0.1 周。

## 敏感性

- q 从 90% 提高到 97.5% 时，各组推荐孕周最多后移 **{max_q_shift:.1f} 周**；BMI 分界不变，因为可靠性水平不参与曲线分组。
- 加入检测误差后，分界点最大变化 **{max_error_cut_shift:.4f} BMI 单位**；估计 pooled within-sample SD 为 **{pooled_sd:.9f}**。
- 在 0.75σ、1.00σ、1.25σ 场景中，推荐孕周相对无误差最多变化 **{max_error_timing_shift:.1f} 周**。

## 审计

- 是否存在平均可靠性≥95%，但组内部分 BMI 低于95%的组：**{'是' if any_below else '否'}**。
- 是否存在最小可靠性低于90%的明显高 BMI 尾部失效：**{'是' if marked_tail else '否'}**。具体 10% 分位、最小值和达标比例见 `final_nipt_timing.xlsx`。

检测误差部分仅为敏感性分析，不表示 Q1 Beta-GAMM 已剥离全部测量误差。本轮未实现 Q3。
"""
        (DECISION_DIR / "q2_summary.md").write_text(summary_md, encoding="utf-8")

        q1_hashes_after = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in (Q1_DATA, Q1_MODEL, SOURCE_XLSX)}
        if q1_hashes_after != input_hashes_before:
            raise AssertionError("Q1 or source attachment changed during Q2")
        manifest = {
            "seed": SEED,
            "q1_hashes": input_hashes_before,
            "q1_conditional_r2": conditional_r2,
            "patient_bmi_definition": "earliest biological observation per patient",
            "n_patients": 267,
            "ga_grid": {"min": 10.0, "max": 25.0, "step": 0.1},
            "min_group_size": MIN_GROUP_SIZE,
            "candidate_K": [2, 3, 4, 5],
            "selected_K": final_k,
            "pooled_measurement_variance": pooled_variance,
            "pooled_measurement_sd": pooled_sd,
            "random_effect_quadrature_nodes": 30,
            "measurement_error_quadrature_nodes": 21,
            "r_runtime_stderr": r_runtime["stderr"],
        }
        (DATA_DIR / "q2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        compact_raw = [
            ("model_info", model_info),
            ("K_summary", segmentation_summary),
            ("segments", segments_all),
            ("timing_all_q", timing_all_q),
            ("error_scenarios", error_scenarios),
            ("error_segments", error_segments),
            ("error_timing", error_timing),
            ("kernel_validation", kernel_validation),
            ("repeat_groups", repeat_groups),
        ]
        specs = [
            {
                "path": "raw/population_reliability_curves.xlsx",
                "sheets": [("patient_curves", population_curves), ("surface_grid", surface_grid), ("patient_profiles", profiles), ("model_info", model_info), ("kernel_validation", kernel_validation)],
            },
            {
                "path": "raw/segmentation_all_K.xlsx",
                "sheets": [("K_summary", segmentation_summary), ("segments_all_K", segments_all), ("error_segments", error_segments)],
            },
            {"path": "raw/timing_all_q.xlsx", "sheets": [("timing_all_q", timing_all_q)]},
            {
                "path": "raw/measurement_error_raw.xlsx",
                "sheets": [
                    ("repeat_groups", repeat_groups),
                    ("error_scenarios", error_scenarios),
                    *[
                        (
                            f"curves_{scenario.replace('measurement_', 'm')}",
                            measurement_curves[measurement_curves["scenario"] == scenario],
                        )
                        for scenario in measurement_sds
                    ],
                    ("error_segments", error_segments),
                    ("error_timing", error_timing),
                ],
            },
            {
                "path": "raw/q2_all_raw.xlsx",
                "sheets": compact_raw,
            },
            {"path": "decision/final_bmi_groups.xlsx", "sheets": [("final_groups", final_groups)]},
            {"path": "decision/final_nipt_timing.xlsx", "sheets": [("final_nipt_timing", final_nipt)]},
            {"path": "decision/sensitivity_summary.xlsx", "sheets": [("sensitivity_summary", sensitivity_summary)]},
        ]
        build_payload(
            [
                {
                    "path": spec["path"],
                    "sheets": [{"name": name, **json_ready(frame)} for name, frame in spec["sheets"]],
                }
                for spec in specs
            ]
        )
        print(f"[5/5] analysis complete: K={final_k}, payload={PAYLOAD_PATH}", flush=True)
    finally:
        resolved_runtime = runtime.resolve()
        if resolved_runtime.parent == args.r_temp.resolve():
            shutil.rmtree(resolved_runtime, ignore_errors=True)


if __name__ == "__main__":
    main()
