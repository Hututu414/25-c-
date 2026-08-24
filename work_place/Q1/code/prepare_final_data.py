from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

Q1_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Q1_ROOT.parent.parent
SOURCE_XLSX = PROJECT_ROOT / "C题" / "附件.xlsx"
OUTPUT_CSV = Q1_ROOT / "data_processed" / "q1_round2_model_data.csv"
MANIFEST_JSON = Q1_ROOT / "data_processed" / "final_data_manifest.json"
SEED = 20260824


def parse_ga(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)\s*[wW]\s*(?:\+\s*(\d+))?\s*", str(value))
    if not match:
        return np.nan
    weeks, days = int(match.group(1)), int(match.group(2) or 0)
    return weeks + days / 7 if 0 <= days <= 6 else np.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare() -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(SOURCE_XLSX, sheet_name="男胎检测数据", engine="openpyxl")
    raw.columns = [str(column).strip() for column in raw.columns]
    required = [
        "孕妇代码", "检测抽血次数", "检测孕周", "Y染色体浓度", "孕妇BMI", "年龄",
        "怀孕次数", "生产次数", "IVF妊娠",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"source workbook misses required columns: {missing}")

    work = raw[required].copy()
    work["GA"] = work["检测孕周"].map(parse_ga)
    for source, target in [
        ("检测抽血次数", "blood_draw_no"),
        ("Y染色体浓度", "Y"),
        ("孕妇BMI", "BMI"),
        ("年龄", "AGE"),
        ("生产次数", "parity"),
    ]:
        work[target] = pd.to_numeric(work[source], errors="coerce")
    if work[["孕妇代码", "blood_draw_no", "GA", "Y", "BMI", "AGE", "parity"]].isna().any().any():
        raise ValueError("required Q1 fields contain missing, invalid, or non-numeric values")
    if not work["Y"].between(0, 1, inclusive="neither").all():
        raise ValueError("Beta-GAMM requires Y strictly inside (0, 1)")

    patients = sorted(work["孕妇代码"].astype(str).unique())
    patient_map = {value: f"P{index:04d}" for index, value in enumerate(patients, 1)}
    work["patient_id"] = work["孕妇代码"].astype(str).map(patient_map)
    work["gravidity_cat"] = work["怀孕次数"].astype(str).str.strip().replace({"≥3": "3plus"})
    if not set(work["gravidity_cat"]).issubset({"1", "2", "3plus"}):
        raise ValueError("unexpected gravidity category")
    allowed_modes = {"自然受孕", "IUI（人工授精）", "IVF（试管婴儿）"}
    if not set(work["IVF妊娠"]).issubset(allowed_modes):
        raise ValueError("unexpected conception mode")
    work["conception_mode"] = np.where(work["IVF妊娠"].eq("自然受孕"), "natural", "assisted")

    keys = ["patient_id", "blood_draw_no", "GA"]
    for column in ["gravidity_cat", "parity", "conception_mode"]:
        if work.groupby(keys, sort=False)[column].nunique(dropna=False).gt(1).any():
            raise AssertionError(f"{column} varies within a technical-repeat group")
    result = (
        work.groupby(keys, as_index=False, sort=True)
        .agg(
            Y=("Y", "mean"), BMI=("BMI", "mean"), AGE=("AGE", "mean"),
            technical_repeat_n=("Y", "size"), gravidity_cat=("gravidity_cat", "first"),
            parity=("parity", "first"), conception_mode=("conception_mode", "first"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    result.insert(0, "row_id", [f"Q1_{index:04d}" for index in range(1, len(result) + 1)])
    result["fold"] = -1
    splitter = GroupKFold(n_splits=5)
    for fold, (_, validation_index) in enumerate(splitter.split(result, groups=result["patient_id"]), 1):
        result.loc[validation_index, "fold"] = fold
    result = result[
        ["row_id", "patient_id", "blood_draw_no", "GA", "Y", "BMI", "AGE",
         "technical_repeat_n", "fold", "gravidity_cat", "parity", "conception_mode"]
    ]

    if result["technical_repeat_n"].sum() != len(raw):
        raise AssertionError("technical-repeat counts do not reconcile to source rows")
    if (result["fold"] < 1).any() or result.groupby("patient_id")["fold"].nunique().max() != 1:
        raise AssertionError("invalid patient-level fold assignment")
    manifest = {
        "source": str(SOURCE_XLSX.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(SOURCE_XLSX),
        "source_rows": len(raw),
        "source_patients": len(patients),
        "aggregated_rows": len(result),
        "technical_repeat_rows_removed": int(len(raw) - len(result)),
        "group_cv": "GroupKFold(n_splits=5), groups=patient_id",
        "seed": SEED,
        "model_data": "q1_round2_model_data.csv",
    }
    return result, manifest


def self_check() -> None:
    assert math.isclose(parse_ga("11w+6"), 11 + 6 / 7)
    assert parse_ga("12W") == 12
    assert np.isnan(parse_ga("bad"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    self_check()
    result, manifest = prepare()
    if args.check_only:
        existing = pd.read_csv(OUTPUT_CSV)
        pd.testing.assert_frame_equal(result, existing, check_dtype=False, rtol=1e-12, atol=1e-12)
        print(f"validated {len(result)} rows and {result['patient_id'].nunique()} patients")
        return
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
