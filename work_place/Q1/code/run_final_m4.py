from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

Q1_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = Q1_ROOT / "data_processed" / "q1_round2_model_data.csv"
MODEL_R = Q1_ROOT / "code" / "m4_round2.R"
TABLE_DIR = Q1_ROOT / "final_results" / "tables"
ARCHIVE_DIAGNOSTICS = Q1_ROOT / "Archive folder" / "diagnostics" / "final_m4_fit"
FINAL_K = 5


def find_rscript() -> Path:
    executable = shutil.which("Rscript")
    if executable:
        return Path(executable)
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = sorted((program_files / "R").glob("R-*/bin/Rscript.exe"), reverse=True)
    if not candidates:
        raise RuntimeError("Rscript not found")
    return candidates[0]


def metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    denominator = np.square(observed - np.mean(observed)).sum()
    r2 = 1.0 - np.square(residual).sum() / denominator
    return float(r2), float(np.sqrt(np.mean(np.square(residual)))), float(np.mean(np.abs(residual)))


def self_check() -> None:
    observed = np.array([1.0, 2.0, 3.0])
    r2, rmse, mae = metrics(observed, observed)
    assert r2 == 1.0 and rmse == 0.0 and mae == 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r-temp", type=Path, required=True)
    args = parser.parse_args()
    self_check()
    r_temp = args.r_temp.resolve()
    if not r_temp.is_absolute() or any(ord(character) > 127 for character in str(r_temp)):
        raise ValueError("--r-temp must be an absolute writable ASCII path")
    r_temp.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_CSV)
    means = {name: float(data[name].mean()) for name in ["GA", "BMI", "AGE"]}
    model_data = data.copy()
    for name, mean in means.items():
        model_data[f"{name}_c"] = model_data[name] - mean
        model_data[f"{name}_mean"] = mean
    columns = [
        "row_id", "patient_id", "GA", "BMI", "AGE", "Y", "gravidity_cat", "parity",
        "conception_mode", "GA_c", "BMI_c", "AGE_c", "GA_mean", "BMI_mean", "AGE_mean",
    ]

    runtime = Path(tempfile.mkdtemp(prefix="q1_final_m4_", dir=r_temp))
    try:
        shutil.copy2(MODEL_R, runtime / MODEL_R.name)
        model_data[columns].to_csv(runtime / "train.csv", index=False)
        model_data[columns].to_csv(runtime / "validation.csv", index=False)
        output_dir = runtime / "output"
        env = os.environ.copy()
        env.update({"TEMP": str(r_temp), "TMP": str(r_temp), "TMPDIR": str(r_temp), "LC_ALL": "C", "LANG": "C"})
        process = subprocess.run(
            [str(find_rscript()), "--vanilla", str(runtime / MODEL_R.name), str(FINAL_K),
             str(runtime / "train.csv"), str(runtime / "validation.csv"), str(output_dir), "1"],
            cwd=runtime, env=env, capture_output=True, shell=False, check=False,
        )
        if process.returncode:
            stderr = process.stderr.decode("utf-8", errors="replace")
            error_file = output_dir / "error.txt"
            error = error_file.read_text(encoding="utf-8", errors="replace") if error_file.exists() else ""
            raise RuntimeError(f"final M4 fit failed: {error or stderr}")

        info = pd.read_csv(output_dir / "info.csv").set_index("key")["value"]
        if str(info.get("converged", "")).upper() not in {"TRUE", "1", "1.0"}:
            raise RuntimeError("final M4 did not converge")
        components = pd.read_csv(output_dir / "training_components.csv")
        fitted = data[["row_id", "patient_id", "Y"]].merge(
            components[["row_id", "mu_conditional"]], on="row_id", how="left", validate="one_to_one"
        )
        if fitted["mu_conditional"].isna().any() or len(fitted) != len(data):
            raise AssertionError("conditional fitted values do not match the full model data")
        r2, rmse, mae = metrics(fitted["Y"].to_numpy(), fitted["mu_conditional"].to_numpy())

        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "Model": "M4 Beta-GAMM",
            "Evaluation": "conditional in-sample fit",
            "Fitted_values": "fixed effects + patient random effects estimated from all observations",
            "N_observations": len(data),
            "N_patients": data["patient_id"].nunique(),
            "k": FINAL_K,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
        }]).to_csv(TABLE_DIR / "final_m4_conditional_fit_metrics.csv", index=False, encoding="utf-8-sig")

        if ARCHIVE_DIAGNOSTICS.exists():
            shutil.rmtree(ARCHIVE_DIAGNOSTICS)
        shutil.copytree(output_dir, ARCHIVE_DIAGNOSTICS)
        fitted.assign(residual=fitted["Y"] - fitted["mu_conditional"]).to_csv(
            ARCHIVE_DIAGNOSTICS / "conditional_predictions.csv", index=False, encoding="utf-8-sig"
        )
        print(f"R2={r2:.9f} RMSE={rmse:.9f} MAE={mae:.9f}")
    finally:
        resolved_runtime = runtime.resolve()
        if resolved_runtime.parent == r_temp:
            shutil.rmtree(resolved_runtime, ignore_errors=True)


if __name__ == "__main__":
    main()
