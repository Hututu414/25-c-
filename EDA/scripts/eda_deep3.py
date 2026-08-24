# -*- coding: utf-8 -*-
"""EDA 补充核对（步骤5）：女胎阈值覆盖、Z~GC 关联、标签行孕周分布"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy import stats

XLSX = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\C题\附件.xlsx")
R = []
def w(*args):
    line = " ".join(str(a) for a in args)
    R.append(line); print(line)

def parse_ga(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"(\d+)\s*[wW]\s*\+\s*(\d+)", str(s).strip())
    return int(m.group(1)) + int(m.group(2)) / 7.0 if m else np.nan

def load_sheet(name, rename=None):
    df = pd.read_excel(XLSX, sheet_name=name)
    if rename:
        df = df.rename(columns=rename)
    df.columns = [str(c).strip() for c in df.columns]
    df["_ga"] = df["检测孕周"].map(parse_ga)
    return df

female = load_sheet("女胎检测数据", {"Unnamed: 20": "Y染色体的Z值", "Unnamed: 21": "Y染色体浓度"})
male = load_sheet("男胎检测数据")
female["_anom"] = female["染色体的非整倍体"].fillna("正常")
male["_anom"] = male["染色体的非整倍体"].fillna("正常")

w("=" * 90)
w("Q. 女胎: 经典阈值 Z>3 与 AB 标签的覆盖关系")
w("=" * 90)
for zc, labels_ab in (("13号染色体的Z值", ["T13", "T13T18", "T13T21"]),
                      ("18号染色体的Z值", ["T18", "T13T18", "T18T21"]),
                      ("21号染色体的Z值", ["T21", "T13T21", "T18T21"])):
    hit = female[female[zc] > 3]
    lab = female[female["_anom"].isin(labels_ab)]
    overlap = hit["_anom"].isin(labels_ab).sum()
    w(f"{zc} > 3: {len(hit)} 行 | 相关AB标签行: {len(lab)} 行 | 命中重叠: {overlap}")
    # 反向: 标签行中 Z>3 的
    w(f"   标签行内 Z>3 比例: {(lab[zc] > 3).mean() * 100:.1f}%  (中位Z={lab[zc].median():.2f})")
w(f"另外: 女胎 21号Z值 全表最大值 = {female['21号染色体的Z值'].max():.2f} (从未超过3)")
w(f"    男胎 21号Z值 全表最大值 = {male['21号染色体的Z值'].max():.2f}")

w("")
w("=" * 90)
w("R. 女胎: Z 值与 指标间相关性 (特征工程参考)")
w("=" * 90)
f = female
pairs = [("21号染色体的Z值", "21号染色体的GC含量"), ("21号染色体的Z值", "GC含量"),
         ("21号染色体的Z值", "X染色体浓度"), ("21号染色体的Z值", "X染色体的Z值"),
         ("21号染色体的Z值", "_ga"), ("21号染色体的Z值", "孕妇BMI"), ("21号染色体的Z值", "原始读段数"),
         ("21号染色体的Z值", "被过滤掉读段数的比例"),
         ("18号染色体的Z值", "18号染色体的GC含量"), ("18号染色体的Z值", "X染色体浓度"),
         ("13号染色体的Z值", "13号染色体的GC含量"), ("13号染色体的Z值", "X染色体浓度"),
         ("13号染色体的Z值", "18号染色体的Z值"), ("21号染色体的Z值", "18号染色体的Z值"),
         ("X染色体的Z值", "X染色体浓度")]
for a, b in pairs:
    sub = f[[a, b]].dropna()
    r, p = stats.spearmanr(sub[a], sub[b])
    w(f"  rho({a}, {b}) = {r:.3f} (p={p:.2e}, n={len(sub)})")

w("")
w("=" * 90)
w("S. 男胎同样的Z~指标关系 (对照)")
w("=" * 90)
m = male
pairs = [("21号染色体的Z值", "21号染色体的GC含量"), ("21号染色体的Z值", "Y染色体浓度"),
         ("21号染色体的Z值", "X染色体浓度"), ("18号染色体的Z值", "18号染色体的GC含量"),
         ("18号染色体的Z值", "X染色体浓度"), ("13号染色体的Z值", "13号染色体的GC含量"),
         ("13号染色体的Z值", "X染色体浓度")]
for a, b in pairs:
    sub = m[[a, b]].dropna()
    r, p = stats.spearmanr(sub[a], sub[b])
    w(f"  rho({a}, {b}) = {r:.3f} (p={p:.2e}, n={len(sub)})")

w("")
w("=" * 90
  )
w("T. AB 标签行的孕周/AGE 分布 (两表)")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    normal = df[df["_anom"] == "正常"]
    abn = df[df["_anom"] != "正常"]
    w(f"[{tag}] 正常行: 孕周 mean={normal['_ga'].mean():.2f}; 异常行: 孕周 mean={abn['_ga'].mean():.2f} (中位 {abn['_ga'].median():.2f})")
    w(f"[{tag}] 异常行 孕周分布: " + str(abn["_ga"].round(0).value_counts().sort_index().to_dict()))
    w(f"[{tag}] 正常行 年龄 mean={normal['年龄'].mean():.1f}, BMI mean={normal['孕妇BMI'].mean():.1f}; 异常行 年龄 mean={abn['年龄'].mean():.1f}, BMI mean={abn['孕妇BMI'].mean():.1f}")
    # 是否健康 与 孕周
    w(f"[{tag}] AE=否 行数: {(df['胎儿是否健康'] == '否').sum()}; 其孕周 mean={df.loc[df['胎儿是否健康'] == '否', '_ga'].mean():.2f}")

with open(Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\EDA\output\deep_analysis3.txt"), "w", encoding="utf-8") as fp:
    fp.write("\n".join(R))
print("saved")
