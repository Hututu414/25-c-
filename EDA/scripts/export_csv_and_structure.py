# -*- coding: utf-8 -*-
"""导出两个 sheet 的 CSV 派生副本（掩码孕妇代码），并输出结构摘要到 UTF-8 文件"""
import pandas as pd
from pathlib import Path
import re

XLSX = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\C题\附件.xlsx")
OUT = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\EDA\output")
OUT.mkdir(parents=True, exist_ok=True)

xl = pd.ExcelFile(XLSX)
print("Sheets:", xl.sheet_names)

def mask_id(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s)
    return re.sub(r"[A-Za-z0-9]+", lambda m: m.group(0)[:2] + "*" * max(len(m.group(0)) - 2, 2), s)

for sh in xl.sheet_names:
    df = pd.read_excel(XLSX, sheet_name=sh)
    tag = "male" if "男" in sh else "female"
    # 掩码孕妇代码列（第2列）
    id_col = df.columns[1]
    df_copy = df.copy()
    df_copy[id_col] = df_copy[id_col].map(mask_id)
    csv_path = OUT / f"sheet_{tag}.csv"
    df_copy.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"{sh}: rows={len(df)}, cols={len(df.columns)} -> {csv_path}")

# 结构摘要
with open(OUT / "excel_structure.md", "w", encoding="utf-8") as f:
    for sh in xl.sheet_names:
        df = pd.read_excel(XLSX, sheet_name=sh)
        f.write(f"\n===== Sheet: {sh} | rows={len(df)} | cols={len(df.columns)} =====\n")
        f.write("columns: " + " | ".join(list(df.columns)) + "\n\n")
        f.write("dtypes:\n")
        for c in df.columns:
            f.write(f"  {c}: {df[c].dtype}\n")
        f.write("\n---- 前3行（掩码） ----\n")
        id_col = df.columns[1]
        tmp = df.head(3).copy()
        tmp[id_col] = tmp[id_col].map(mask_id)
        f.write(tmp.to_string() + "\n")

print("done")
