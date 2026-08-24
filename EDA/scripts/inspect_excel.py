# -*- coding: utf-8 -*-
"""检查附件 Excel 结构：sheet 列表、列名、类型、行数"""
import pandas as pd
from pathlib import Path

xlsx = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\C题\附件.xlsx")
xl = pd.ExcelFile(xlsx)
print("Sheets:", xl.sheet_names)
for sh in xl.sheet_names:
    df = pd.read_excel(xlsx, sheet_name=sh, nrows=5)
    df_full = pd.read_excel(xlsx, sheet_name=sh)
    print(f"\n===== Sheet: {sh} | rows={len(df_full)} | cols={len(df_full.columns)} =====")
    print("columns:", list(df_full.columns))
    # 简单类型信息
    print("dtypes:")
    for c in df_full.columns:
        print(f"  {c}: {df_full[c].dtype}")
