from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

Q4_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Q4_ROOT.parents[1]
SOURCE_XLSX = PROJECT_ROOT / "C题" / "附件.xlsx"
DATA_DIR = Q4_ROOT / "data_processed"


def parse_ga(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\s*(\d+)\s*[wW]\s*(?:\+\s*(\d+))?\s*", str(value))
    if not match:
        return np.nan
    weeks, days = int(match.group(1)), int(match.group(2) or 0)
    return weeks + days / 7 if 0 <= days <= 6 else np.nan


def prepare_data() -> tuple[pd.DataFrame, dict[str, object]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(SOURCE_XLSX, sheet_name=1, engine="openpyxl")
    raw.columns = [str(column).strip() for column in raw.columns]
    c = raw.columns
    patients = sorted(raw[c[1]].astype(str).unique())
    patient_map = {value: f"P{index:04d}" for index, value in enumerate(patients, 1)}
    ab = raw[c[27]].fillna("").astype(str).str.strip()

    data = pd.DataFrame(
        {
            "row_id": pd.to_numeric(raw[c[0]], errors="coerce").astype("Int64"),
            "patient_id": raw[c[1]].astype(str).map(patient_map),
            "age": pd.to_numeric(raw[c[2]], errors="coerce"),
            "blood_draw_no": pd.to_numeric(raw[c[8]], errors="coerce"),
            "GA": raw[c[9]].map(parse_ga),
            "BMI": pd.to_numeric(raw[c[10]], errors="coerce"),
            "raw_reads": pd.to_numeric(raw[c[11]], errors="coerce"),
            "alignment_rate": pd.to_numeric(raw[c[12]], errors="coerce"),
            "duplication_rate": pd.to_numeric(raw[c[13]], errors="coerce"),
            "unique_reads": pd.to_numeric(raw[c[14]], errors="coerce"),
            "GC_global": pd.to_numeric(raw[c[15]], errors="coerce"),
            "Z13": pd.to_numeric(raw[c[16]], errors="coerce"),
            "Z18": pd.to_numeric(raw[c[17]], errors="coerce"),
            "Z21": pd.to_numeric(raw[c[18]], errors="coerce"),
            "X_Z": pd.to_numeric(raw[c[19]], errors="coerce"),
            "X_conc": pd.to_numeric(raw[c[22]], errors="coerce"),
            "GC13": pd.to_numeric(raw[c[23]], errors="coerce"),
            "GC18": pd.to_numeric(raw[c[24]], errors="coerce"),
            "GC21": pd.to_numeric(raw[c[25]], errors="coerce"),
            "filtered_rate": pd.to_numeric(raw[c[26]], errors="coerce"),
            "AB_label": ab,
        }
    )
    for chromosome in (13, 18, 21):
        data[f"true_T{chromosome}"] = ab.str.contains(f"T{chromosome}", regex=False).astype(int)
    data["true_any"] = data[["true_T13", "true_T18", "true_T21"]].max(axis=1)
    data["is_normal_reference"] = ab.eq("").astype(int)
    data["log_raw_reads"] = np.log1p(data["raw_reads"])
    data["log_unique_reads"] = np.log1p(data["unique_reads"])

    repeat_key = list(
        zip(
            data["patient_id"],
            data["blood_draw_no"].fillna(-1),
            data["GA"].round(8).fillna(-1),
            strict=True,
        )
    )
    unique_keys = {key: f"TR{index:04d}" for index, key in enumerate(sorted(set(repeat_key)), 1)}
    data["technical_repeat_group_id"] = [unique_keys[key] for key in repeat_key]

    required = [
        "row_id", "patient_id", "blood_draw_no", "GA", "age", "raw_reads",
        "alignment_rate", "duplication_rate", "unique_reads", "GC_global",
        "Z13", "Z18", "Z21", "X_Z", "X_conc", "GC13", "GC18", "GC21",
        "filtered_rate",
    ]
    if data[required].isna().any().any():
        missing = data[required].isna().sum()
        raise ValueError(f"unexpected missing required values: {missing[missing.gt(0)].to_dict()}")
    if data["BMI"].isna().sum() > 1:
        raise ValueError("unexpected BMI missingness")
    expected = {13: 23, 18: 46, 21: 13}
    observed = {chromosome: int(data[f"true_T{chromosome}"].sum()) for chromosome in expected}
    if len(data) != 605 or data["patient_id"].nunique() != 147 or observed != expected:
        raise AssertionError(
            f"female-data contract changed: rows={len(data)}, patients={data['patient_id'].nunique()}, labels={observed}"
        )

    audit = {
        "source": str(SOURCE_XLSX.relative_to(PROJECT_ROOT)),
        "sheet": "female sheet (index 1)",
        "rows": len(data),
        "patients": int(data["patient_id"].nunique()),
        "normal_rows": int(data["is_normal_reference"].sum()),
        "abnormal_rows": int(data["true_any"].sum()),
        "positive_rows": {f"T{chromosome}": observed[chromosome] for chromosome in expected},
        "positive_patients": {
            f"T{chromosome}": int(data.loc[data[f"true_T{chromosome}"].eq(1), "patient_id"].nunique())
            for chromosome in expected
        },
        "BMI_missing": int(data["BMI"].isna().sum()),
        "label_contract": "AB is the task-specified observed decision label, not a verified karyotype truth",
        "AE_used": False,
        "male_rows_used": 0,
    }
    data.to_csv(DATA_DIR / "q4_model_data.csv", index=False, encoding="utf-8-sig")
    (DATA_DIR / "q4_data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data, audit


if __name__ == "__main__":
    frame, report = prepare_data()
    print(f"prepared rows={len(frame)}, patients={report['patients']}, abnormal={report['abnormal_rows']}")
