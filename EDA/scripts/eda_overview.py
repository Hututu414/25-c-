# -*- coding: utf-8 -*-
"""
EDA 总览脚本（步骤1）：
- 载入男胎/女胎两个 sheet（原始 xlsx 只读）
- 统一列名，解析孕周/日期
- 输出：唯一孕妇数、重复测量结构、分类取值、缺失表、数值概要、BMI一致性、孕周一致性
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

XLSX = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\C题\附件.xlsx")
OUT = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\EDA\output")
R = []  # report lines

def w(*args):
    line = " ".join(str(a) for a in args)
    R.append(line)
    print(line)

def parse_ga(s):
    """'11w+6' -> 11 + 6/7 周"""
    if pd.isna(s):
        return np.nan
    m = re.match(r"(\d+)w\+(\d+)", str(s).strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / 7.0
    m2 = re.match(r"(\d+)w", str(s).strip())
    if m2:
        return float(m2.group(1))
    return np.nan

def parse_date(x):
    """int 20230429 或 '2023-04-29 00:00:00' -> date"""
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float)):
        s = str(int(x))
        if len(s) == 8:
            try:
                return datetime.strptime(s, "%Y%m%d")
            except ValueError:
                return pd.NaT
    s = str(x).strip()
    if re.match(r"^\d{8}$", s):
        return datetime.strptime(s, "%Y%m%d")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return pd.NaT

def load_sheet(name, rename=None):
    df = pd.read_excel(XLSX, sheet_name=name)
    if rename:
        df = df.rename(columns=rename)
    # 保持中文列名，统一关键列
    df.columns = [str(c).strip() for c in df.columns]
    # 去除列名尾部空格（"唯一比对的读段数  "）
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df

rename_f = {"Unnamed: 20": "Y染色体的Z值", "Unnamed: 21": "Y染色体浓度"}
male = load_sheet("男胎检测数据")
female = load_sheet("女胎检测数据", rename_f)

w("=" * 90)
w("一、基本规模与唯一性")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    w(f"[{tag}] 行数={len(df)}, 列数={len(df.columns)}")
    ids = df["孕妇代码"]
    w(f"[{tag}] 唯一孕妇数 = {ids.nunique()}")
    vc = ids.value_counts()
    w(f"[{tag}] 每名孕妇检测次数分布: min={vc.min()}, max={vc.max()}, mean={vc.mean():.2f}, median={vc.median():.0f}")
    w(f"[{tag}] 检测次数=1的孕妇数: {(vc == 1).sum()}, 检测次数>=2: {(vc >= 2).sum()}, >=3: {(vc >= 3).sum()}, >=4: {(vc >= 4).sum()}, >=5: {(vc >= 5).sum()}")
    # 孕妇代码是否跨表重复
male_ids = set(male["孕妇代码"])
female_ids = set(female["孕妇代码"])
w(f"[跨表] 男胎/女胎孕妇代码交集 = {len(male_ids & female_ids)} (0 表示两表孕妇互不相同)")
w(f"[跨表] 男胎专属={len(male_ids - female_ids)}, 女胎专属={len(female_ids - male_ids)}")
w(f"[合计] 孕妇总数 = {len(male_ids | female_ids)}")

# 序号检查
for tag, df in (("男胎", male), ("女胎", female)):
    s = df["序号"]
    w(f"[{tag}] 序号 min={s.min()}, max={s.max()}, 唯一={s.nunique()}, 从1连续={list(s)==list(range(1, len(s)+1))}")

w("")
w("=" * 90)
w("二、分类字段取值（计数）")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    w(f"\n--- [{tag}] ---")
    for col in ["IVF妊娠", "染色体的非整倍体", "胎儿是否健康", "怀孕次数", "检测抽血次数"]:
        w(f"[{col}] 取值: {df[col].value_counts(dropna=False).to_dict()}")

w("")
w("=" * 90)
w("三、缺失统计")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    w(f"\n--- [{tag}] ---")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss) == 0:
        w("所有列均无缺失")
    for c, n in miss.items():
        w(f"  {c}: {n} ({n / len(df) * 100:.2f}%)")

w("")
w("=" * 90)
w("四、日期/孕周解析与一致性")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    df["_ga"] = df["检测孕周"].map(parse_ga)
    df["_tdate"] = df["检测日期"].map(parse_date)
    df["_lmp"] = df["末次月经"].map(parse_date)
    w(f"\n--- [{tag}] ---")
    w(f"孕周解析成功: {df['_ga'].notna().sum()}/{len(df)}; 检测日期解析成功: {df['_tdate'].notna().sum()}/{len(df)}; 末次月经非空: {df['_lmp'].notna().sum()}/{len(df)}")
    # 孕周范围
    w(f"检测孕周(decimal): min={df['_ga'].min():.2f}, max={df['_ga'].max():.2f}, mean={df['_ga'].mean():.2f}")
    # 日期范围
    w(f"检测日期范围: {df['_tdate'].min()} ~ {df['_tdate'].max()}")
    lmp_nonnull = df[df["_lmp"].notna()]
    w(f"末次月经日期范围: {lmp_nonnull['_lmp'].min()} ~ {lmp_nonnull['_lmp'].max()}")
    # 由末次月经推算孕周 = (检测日期 - 末次月经)/7
    sub = df[df["_tdate"].notna() & df["_lmp"].notna()].copy()
    sub["_ga_calc"] = (sub["_tdate"] - sub["_lmp"]).dt.days / 7.0
    sub["_diff"] = (sub["_ga"] - sub["_ga_calc"]).abs()
    w(f"按末次月经推算孕周与记录孕周之差(|周|): mean={sub['_diff'].mean():.3f}, 中位={sub['_diff'].median():.3f}, 95分位={sub['_diff'].quantile(0.95):.3f}, max={sub['_diff'].max():.3f}")
    w(f"|差| < 0.3周: {(sub['_diff'] < 0.3).sum() / len(sub) * 100:.1f}%;  >= 1周: {(sub['_diff'] >= 1).sum()} 行;  >= 3周: {(sub['_diff'] >= 3).sum()} 行")
    # 同一孕妇的孕周是否随时间递增（按检测抽血次数排序）
    inc_ok, inc_bad, n_pairs = 0, 0, 0
    for pid, g in df.groupby("孕妇代码"):
        g = g.sort_values("检测抽血次数")
        g = g.dropna(subset=["_ga"])
        if len(g) < 2:
            continue
        prev = None
        for v in g["_ga"]:
            if prev is not None:
                n_pairs += 1
                if v > prev - 1e-9:
                    inc_ok += 1
                else:
                    inc_bad += 1
            prev = v
    w(f"同一孕妇相邻检测孕周递增次数: {inc_ok}, 倒退/持平: {inc_bad} / 共 {n_pairs} 对")

w("")
w("=" * 90)
w("五、BMI 一致性（BMI = 体重 / 身高^2, 身高m）")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    calc = df["体重"] / (df["身高"] / 100.0) ** 2
    diff = (calc - df["孕妇BMI"]).abs()
    w(f"[{tag}] BMI 字段与 体重/身高^2 之差的绝对值: mean={diff.mean():.4f}, max={diff.max():.4f}, 一致率(<0.01)={(diff < 0.01).mean() * 100:.2f}%")

w("")
w("=" * 90)
w("六、数值字段概要 (mean / std / min / p25 / p50 / p75 / max)")
w("=" * 90)
num_cols = ["年龄", "身高", "体重", "孕妇BMI", "原始读段数", "在参考基因组上比对的比例",
            "重复读段的比例", "唯一比对的读段数", "GC含量", "13号染色体的Z值", "18号染色体的Z值",
            "21号染色体的Z值", "X染色体的Z值", "Y染色体的Z值", "Y染色体浓度", "X染色体浓度",
            "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
            "被过滤掉读段数的比例", "怀孕次数", "生产次数"]
for tag, df in (("男胎", male), ("女胎", female)):
    w(f"\n--- [{tag}] ---")
    for c in num_cols:
        if c not in df.columns:
            w(f"  {c}: 列不存在")
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        w(f"  {c}: n={v.notna().sum()} mean={v.mean():.4g} std={v.std():.4g} min={v.min():.4g} p25={v.quantile(0.25):.4g} med={v.median():.4g} p75={v.quantile(0.75):.4g} max={v.max():.4g}")

with open(OUT / "overview_stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("\nsaved ->", OUT / "overview_stats.txt")
