from __future__ import annotations

import argparse
import shutil
from pathlib import Path

Q1_ROOT = Path(__file__).resolve().parents[3]

MOVES = {
    "README.md": "Archive folder/diagnostics/reports/README_before_cleanup.md",
    "code/gamm_models.R": "Archive folder/first_round/code/gamm_models.R",
    "code/run_q1_models.py": "Archive folder/first_round/code/run_q1_models.py",
    "code/run_round2.py": "Archive folder/diagnostics/code/run_round2.py",
    "code/build_round2_workbooks.mjs": "Archive folder/diagnostics/code/build_round2_workbooks.mjs",
    "data_processed/q1_male_aggregated.csv": "Archive folder/first_round/data_processed/q1_male_aggregated.csv",
    "data_processed/fold_assignments.csv": "Archive folder/first_round/data_processed/fold_assignments.csv",
    "data_processed/data_manifest.json": "Archive folder/first_round/data_processed/data_manifest.json",
    "data_processed/round2_manifest.json": "Archive folder/diagnostics/data/round2_manifest.json",
    "outputs/decision/decision_summary.md": "Archive folder/first_round/decision/decision_summary.md",
    "outputs/decision/model_decision_table.csv": "Archive folder/first_round/decision/model_decision_table.csv",
    "outputs/decision/model_decision_table.xlsx": "Archive folder/first_round/decision/model_decision_table.xlsx",
    "outputs/figures/model_cv_comparison.png": "Archive folder/first_round/figures/model_cv_comparison.png",
    "outputs/figures/winner_effects.png": "Archive folder/first_round/figures/winner_effects.png",
    "outputs/figures/winner_pred_vs_obs.png": "Archive folder/first_round/figures/winner_pred_vs_obs.png",
    "outputs/raw/all_models_raw.xlsx": "Archive folder/first_round/raw/all_models_raw.xlsx",
    "outputs/raw/B0_raw.xlsx": "Archive folder/first_round/raw/B0_raw.xlsx",
    "outputs/raw/M1_raw.xlsx": "Archive folder/first_round/raw/M1_raw.xlsx",
    "outputs/raw/M2_raw.xlsx": "Archive folder/first_round/raw/M2_raw.xlsx",
    "outputs/raw/M3_raw.xlsx": "Archive folder/first_round/raw/M3_raw.xlsx",
    "outputs/raw/M4_raw.xlsx": "Archive folder/first_round/raw/M4_raw.xlsx",
    "outputs/raw/M5_raw.xlsx": "Archive folder/first_round/raw/M5_raw.xlsx",
    "outputs_round2/raw/unseen_B0_raw.xlsx": "Archive folder/unseen_patient/raw/unseen_B0_raw.xlsx",
    "outputs_round2/raw/unseen_M4P_raw.xlsx": "Archive folder/unseen_patient/raw/unseen_M4P_raw.xlsx",
    "outputs_round2/raw/unseen_SGEE_raw.xlsx": "Archive folder/unseen_patient/raw/unseen_SGEE_raw.xlsx",
    "outputs_round2/raw/m4_tuning_raw.xlsx": "Archive folder/unseen_patient/tuning/m4_tuning_raw.xlsx",
    "outputs_round2/decision/unseen_decision_table.csv": "Archive folder/unseen_patient/decision/unseen_decision_table.csv",
    "outputs_round2/decision/unseen_decision_table.xlsx": "Archive folder/unseen_patient/decision/unseen_decision_table.xlsx",
    "outputs_round2/figures/unseen_model_comparison.png": "Archive folder/unseen_patient/figures/unseen_model_comparison.png",
    "outputs_round2/raw/seen_M4C_raw.xlsx": "Archive folder/seen_patient_validation/raw/seen_M4C_raw.xlsx",
    "outputs_round2/decision/seen_decision_table.csv": "Archive folder/seen_patient_validation/decision/seen_decision_table.csv",
    "outputs_round2/decision/seen_decision_table.xlsx": "Archive folder/seen_patient_validation/decision/seen_decision_table.xlsx",
    "outputs_round2/decision/personalization_gain.csv": "Archive folder/seen_patient_validation/decision/personalization_gain.csv",
    "outputs_round2/decision/personalization_gain.xlsx": "Archive folder/seen_patient_validation/decision/personalization_gain.xlsx",
    "outputs_round2/figures/m4_population_vs_conditional.png": "Archive folder/seen_patient_validation/figures/m4_population_vs_conditional.png",
    "outputs_round2/figures/personalization_gain.png": "Archive folder/seen_patient_validation/figures/personalization_gain.png",
    "outputs_round2/raw/round2_all_raw.xlsx": "Archive folder/diagnostics/combined/round2_all_raw.xlsx",
    "outputs_round2/decision/decision_summary.md": "Archive folder/diagnostics/reports/round2_decision_summary.md",
    "outputs_round2/figures/m4_effect_surface.png": "final_results/figures/m4_effect_surface.png",
}


def checked_path(relative: str) -> Path:
    path = (Q1_ROOT / relative).resolve()
    if Q1_ROOT.resolve() not in path.parents:
        raise ValueError(f"path escaped Q1: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pairs = [(checked_path(source), checked_path(destination)) for source, destination in MOVES.items()]
    missing = [str(source) for source, _ in pairs if not source.is_file()]
    existing = [str(destination) for _, destination in pairs if destination.exists()]
    if missing or existing:
        raise RuntimeError(f"missing sources={missing}; existing destinations={existing}")
    if args.dry_run:
        print(f"validated {len(pairs)} moves inside {Q1_ROOT}")
        return
    for source, destination in pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
        print(f"{source.relative_to(Q1_ROOT)} -> {destination.relative_to(Q1_ROOT)}")
    for relative in [
        "outputs/raw", "outputs/decision", "outputs/figures", "outputs",
        "outputs_round2/raw", "outputs_round2/decision", "outputs_round2/figures", "outputs_round2",
    ]:
        directory = Q1_ROOT / relative
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


if __name__ == "__main__":
    main()
