# -*- coding: utf-8 -*-
"""
EDA 深度分析（步骤3）：
G. 男胎按 AB 标签的 Z 值区分度（AUC）与阈值对应关系
H. AB 标签 vs 测序质量特征（两表）
I. 同次采血多次检测的测量噪声（检测误差量化）
J. 女胎 X 染色体 Z 值/浓度 极端行检查
K. 孕周格式/未解析行检查; IVF/怀孕次数 孕妇级恒定检查
L. 孕妇级 最早达标周 ~ 体重/身高/年龄/孕次 (问题3 预分析)
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import roc_auc_score

XLSX = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\C题\附件.xlsx")
OUT = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\EDA\output")
R = []

def w(*args):
    line = " ".join(str(a) for a in args)
    R.append(line)
    print(line)

def parse_ga(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"(\d+)w\+(\d+)", str(s).strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / 7.0
    m2 = re.match(r"(\d+)w", str(s).strip())
    return float(m2.group(1)) if m2 else np.nan

def load_sheet(name, rename=None):
    df = pd.read_excel(XLSX, sheet_name=name)
    if rename:
        df = df.rename(columns=rename)
    df.columns = [str(c).strip() for c in df.columns]
    df["_ga"] = df["检测孕周"].map(parse_ga)
    return df

female = load_sheet("女胎检测数据", {"Unnamed: 20": "Y染色体的Z值", "Unnamed: 21": "Y染色体浓度"})
male = load_sheet("男胎检测数据")
for df in (male, female):
    df["_anom"] = df["染色体的非整倍体"].fillna("正常")
QUAL = ["在参考基因组上比对的比例", "重复读段的比例", "被过滤掉读段数的比例", "GC含量",
        "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "原始读段数"]

w("=" * 90)
w("G. 男胎: AB 标签与 Z 值对应性")
w("=" * 90)
w("男胎 AB 类别计数: " + male["_anom"].value_counts().to_dict().__str__())
for label, chrom in (("T13", "13"), ("T18", "18"), ("T21", "21")):
    zc = f"{chrom}号染色体的Z值"
    sub = male[male["_anom"].isin(["正常", label])]
    yb = (sub["_anom"] == label).astype(int)
    auc = roc_auc_score(yb, sub[zc])
    w(f"[男胎] {label} vs 正常: n={len(sub)}, {zc} AUC={auc:.3f}, "
      f"标签行{zc}: mean={sub[sub['_anom']==label][zc].mean():.2f}, "
      f"正常行{zc}: mean={sub[sub['_anom']=='正常'][zc].mean():.2f}")
w("男胎 各 AB 标签行 的 Z 值统计 (mean ± sd):")
for label in male["_anom"].unique():
    sub = male[male["_anom"] == label]
    zs = [f"{c}={sub[c].mean():.2f}±{sub[c].std():.2f}" for c in
          ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "Y染色体的Z值"]]
    w(f"  {label} (n={len(sub)}): " + "; ".join(zs))
w("男胎 各行中 标签染色体 Z 值 > 3 的比例:")
for label, chrom in (("T13", "13"), ("T18", "18"), ("T21", "21"), ("T13T18", "13/18")):
    sub = male[male["_anom"] == label]
    if chrom == "13/18":
        hit = ((sub["13号染色体的Z值"] > 3) | (sub["18号染色体的Z值"] > 3)).mean()
    else:
        hit = (sub[f"{chrom}号染色体的Z值"] > 3).mean()
    w(f"  {label}: 标签染色体Z>3比例 = {hit * 100:.1f}% (n={len(sub)})")
w("男胎 各 AB 标签 对应染色体 Z 的分位数:")
for label, chrom in (("T13", "13"), ("T18", "18"), ("T21", "21")):
    sub = male[male["_anom"] == label][f"{chrom}号染色体的Z值"]
    w(f"  {label}: p25={sub.quantile(0.25):.2f} med={sub.median():.2f} p75={sub.quantile(0.75):.2f} p90={sub.quantile(0.9):.2f} p95={sub.quantile(0.95):.2f} max={sub.max():.2f}")

w("")
w("=" * 90)
w("H. AB 标签 与 测序质量特征")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    w(f"\n--- [{tag}] 各标签质量特征均值 ---")
    q = df.groupby("_anom")[QUAL].mean()
    w(q.round(4).to_string())
    # "组合标签" vs 单纯标签 的质量对比
    combo = df[df["_anom"].str.contains(r"T\d+T\d+", na=False)]
    single = df[df["_anom"].isin(["T13", "T18", "T21"])]
    w(f"  组合标签行数={len(combo)}; 过滤比例组合={combo['被过滤掉读段数的比例'].mean():.4f} vs 单纯标签={single['被过滤掉读段数的比例'].mean():.4f} vs 正常={df[df['_anom']=='正常']['被过滤掉读段数的比例'].mean():.4f}")

w("")
w("=" * 90)
w("I. 同次采血多次检测 -> 测量噪声")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    w(f"\n--- [{tag}] ---")
    for col in ["Y染色体浓度", "X染色体浓度", "13号染色体的Z值", "18号染色体的Z值",
                "21号染色体的Z值", "X染色体的Z值", "GC含量", "原始读段数"]:
        sds = []
        for (pid, draw), g in df.groupby(["孕妇代码", "检测抽血次数"]):
            if len(g) >= 2:
                sds.append(g[col].std())
        if sds:
            sds = np.array(sds)
            rel = sds / (df[col].abs().mean() * 100)  # 相对粗尺度
            w(f"  {col}: 组合数(>=2次检测)={len(sds)}, 组内SD均值={sds.mean():.5g}, SD中位={np.median(sds):.5g}, SD最大={sds.max():.5g}")
    # Y浓度组内SD相对均值
    sds = []
    for (pid, draw), g in df.groupby(["孕妇代码", "检测抽血次数"]):
        if len(g) >= 2:
            sds.append(g["Y染色体浓度"].std())
    if sds:
        sds = np.array(sds)
        w(f"  Y浓度 组内SD/全表均值 = {sds.mean() / df['Y染色体浓度'].mean() * 100:.1f}%")

w("")
w("=" * 90)
w("J. 女胎 X Z值/浓度 极端行")
w("=" * 90)
f = female
for thr in (3.0, 5.0):
    sub = f[f["X染色体的Z值"].abs() > thr]
    w(f"|X Z值| > {thr}: {len(sub)} 行; 其中AB标签: {sub['_anom'].value_counts().to_dict()}; "
      f"X浓度 mean={sub['X染色体浓度'].mean():.4f}; 孕周 mean={sub['_ga'].mean():.1f}")
    sub2 = f[f["X染色体浓度"].abs() > 0.06]
    w(f"|X浓度| > 0.06: {len(sub2)} 行; AB标签: {sub2['_anom'].value_counts().to_dict()}; 孕周 mean={sub2['_ga'].mean():.1f}")
# X Z值 极值行的孕周/浓度分布
w("\n女胎 X染色体Z值 全表: p1={:.2f} p5={:.2f} p95={:.2f} p99={:.2f}".format(
    f["X染色体的Z值"].quantile(0.01), f["X染色体的Z值"].quantile(0.05),
    f["X染色体的Z值"].quantile(0.95), f["X染色体的Z值"].quantile(0.99)))

w("")
w("=" * 90)
w("K. 异常值/格式检查")
w("=" * 90)
w("男胎 检测孕周 未解析的原始字符串(仅类别): " + str(male[male["_ga"].isna()]["检测孕周"].unique().tolist()))
w("孕周字符串格式去重(前20): " + str(sorted(male["检测孕周"].unique())[:20]))
w("女胎 BMI 缺失行: 孕周={}, 年龄={}".format(female.loc[female["孕妇BMI"].isna(), "检测孕周"].tolist(),
                                        female.loc[female["孕妇BMI"].isna(), "年龄"].tolist()))
w("男胎 末次月经缺失 12 行 的孕妇检测抽血次数: {}".format(male.loc[male["末次月经"].isna(), "检测抽血次数"].tolist()))
# IVF/怀孕次数 孕妇级恒定
for tag, df in (("男胎", male), ("女胎", female)):
    for col in ["IVF妊娠", "怀孕次数"]:
        n_uni = df.groupby("孕妇代码")[col].nunique()
        w(f"[{tag}] {col} 孕妇内恒定: {(n_uni == 1).sum()}/{len(n_uni)}, 变化: {(n_uni > 1).sum()}")

w("")
w("=" * 90)
w("L. 孕妇级: 最早达标周 与 身高/体重/年龄/孕次 的关系 (问题3 预分析)")
w("=" * 90)
first = male.groupby("孕妇代码").first().copy()
fo = {}
for pid, g in male.groupby("孕妇代码"):
    g = g[g["Y染色体浓度"] >= 0.04]
    fo[pid] = g["_ga"].min() if len(g) else np.nan
first["_firstwk"] = first.index.map(fo)
sub = first.dropna(subset=["_firstwk"])
w(f"可分析孕妇(曾达标) = {len(sub)}")
for c in ["孕妇BMI", "年龄", "身高", "体重", "生产次数"]:
    r, p = stats.spearmanr(sub[c], sub["_firstwk"])
    w(f"  最早达标周 vs {c}: rho={r:.3f} (p={p:.2e})")
# 多重回归 (OLS)
try:
    import statsmodels.api as sm
    X = sm.add_constant(sub[["孕妇BMI", "年龄", "身高"]])
    m = sm.OLS(sub["_firstwk"], X).fit()
    w("\n  OLS firstwk ~ BMI + 年龄 + 身高 (n={}):".format(len(sub)))
    w("  params: " + {k: round(v, 3) for k, v in m.params.items()}.__str__())
    w("  pvalues: " + {k: round(v, 4) for k, v in m.pvalues.items()}.__str__())
    w(f"  R2={m.rsquared:.3f} adjR2={m.rsquared_adj:.3f}")
    w(f"  VIF: " + {k: round(v, 2) for k, v in sm.stats.outliers_influence.variance_inflation_factor(
        sm.add_constant(sub[["孕妇BMI", "年龄", "身高"]].values), 1).items()}.__str__())
except Exception as e:
    w("OLS 失败:", e)
# BMI 与 体重/身高 相关
r1, _ = stats.spearmanr(first["孕妇BMI"], first["体重"])
r2, _ = stats.spearmanr(first["孕妇BMI"], first["身高"])
w(f"  BMI-体重 rho={r1:.3f}; BMI-身高 rho={r2:.3f}")
# 从未达标与已达标孕妇的特征对比
never = first[first["_firstwk"].isna()]
ever = first[first["_firstwk"].notna()]
for c in ["孕妇BMI", "年龄", "体重", "身高"]:
    w(f"  {c}: 达标本({ever[c].mean():.2f}) vs 未达标本({never[c].mean():.2f})")

with open(OUT / "deep_analysis2.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(R))
print("\nsaved ->", OUT / "deep_analysis2.txt")
