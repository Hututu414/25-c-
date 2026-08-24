# -*- coding: utf-8 -*-
"""
EDA 分析（步骤4）：达标率曲线、检测误差影响、ABxAE 关联、早达标子集分析
输出: output/model_analysis.txt
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy import stats
from scipy.optimize import brentq

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
    m = re.match(r"(\d+)\s*[wW]\s*\+\s*(\d+)", str(s).strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / 7.0
    m2 = re.match(r"(\d+)\s*[wW]", str(s).strip())
    return float(m2.group(1)) if m2 else np.nan

def load_sheet(name, rename=None):
    df = pd.read_excel(XLSX, sheet_name=name)
    if rename:
        df = df.rename(columns=rename)
    df.columns = [str(c).strip() for c in df.columns]
    df["_ga"] = df["检测孕周"].map(parse_ga)
    return df

male = load_sheet("男胎检测数据")
female = load_sheet("女胎检测数据", {"Unnamed: 20": "Y染色体的Z值", "Unnamed: 21": "Y染色体浓度"})
male["_anom"] = male["染色体的非整倍体"].fillna("正常")
female["_anom"] = female["染色体的非整倍体"].fillna("正常")
bins = [20, 28, 32, 36, 40, np.inf]
labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "40+"]
male["_bmig"] = pd.cut(male["孕妇BMI"], bins=bins, labels=labels, right=False)

w("=" * 90)
w("M. 达标率曲线: logit P(>=4%) ~ 孕周, 按BMI组")
w("=" * 90)
import statsmodels.api as sm
male["_ok"] = (male["Y染色体浓度"] >= 0.04).astype(int)
rows = []
for lab in labels:
    s = male[male["_bmig"] == lab].dropna(subset=["_ga"])
    if len(s) < 15:
        continue
    X = sm.add_constant(s["_ga"])
    m = sm.GLM(s["_ok"], X, family=sm.families.Binomial()).fit()
    a, b = m.params["const"], m.params["_ga"]
    se_a, se_b = m.bse["const"], m.bse["_ga"]
    def p(ga):
        z = a + b * ga
        return 1 / (1 + np.exp(-z))
    def solve_p(p0, sgn):
        f = lambda g: p(g) - p0
        return brentq(f, 10, 35) if sgn * (p(10) - p0) > 0 or sgn > 0 else np.nan
    t50 = brentq(lambda g: p(g) - 0.5, 10, 30) if (p(10) - 0.5) * (p(30) - 0.5) < 0 else np.nan
    t80 = brentq(lambda g: p(g) - 0.8, 10, 30) if (p(10) - 0.8) * (p(30) - 0.8) < 0 else np.nan
    t90 = brentq(lambda g: p(g) - 0.9, 10, 30) if (p(10) - 0.9) * (p(30) - 0.9) < 0 else np.nan
    t95 = brentq(lambda g: p(g) - 0.95, 10, 30) if (p(10) - 0.95) * (p(30) - 0.95) < 0 else np.nan
    w(f"[{lab}] n={len(s)}: a={a:.3f}(±{se_a:.3f}) b={b:.4f}(±{se_b:.4f}) R2={m.pearson_chi2 / m.nobs if hasattr(m,'pearson_chi2') else np.nan:.3f}")
    w(f"       P(达标) = {p(12) * 100:.1f}% @12w, {p(13) * 100:.1f}% @13w, {p(14) * 100:.1f}% @14w, {p(16) * 100:.1f}% @16w, {p(20) * 100:.1f}% @20w")
    w(f"       达到 50%/80%/90%/95% 的孕周: {t50:.2f} / {t80:.2f} / {t90:.2f} / {t95:.2f}")
    rows.append([lab, len(s), a, b])
    # 众数法: 各GA达标率的观测值保留在表格
# 全局模型（带 BMI 交互）
s = male.dropna(subset=["_ga"])
X = sm.add_constant(s[["_ga", "孕妇BMI", "孕妇BMI"]])
X.columns = ["const", "_ga", "孕妇BMI", "gaxbmi"]
X["gaxbmi"] = s["_ga"] * s["孕妇BMI"]
m = sm.GLM(s["_ok"], X, family=sm.families.Binomial()).fit()
w("\n全局 logit: 达标 ~ GA + BMI + GA*BMI")
w("  params: " + {k: round(v, 4) for k, v in m.params.items()}.__str__())
w("  pvalues: " + {k: round(v, 4) for k, v in m.pvalues.items()}.__str__())

w("")
w("=" * 90)
w("N. 检测误差影响（使用同次采血重复检测的组内SD）")
w("=" * 90)
sigma_y = 0.005441  # 男胎 Y浓度 组内SD均值
sub = male["Y染色体浓度"]
w(f"同次采血重复检测: Y浓度 组内SD 均值 = {sigma_y:.5f}")
w(f"落在 4% 阈值 ±1σ=({0.04 - sigma_y:.4f}, {0.04 + sigma_y:.4f}) 区间内的行数 = {((sub > 0.04 - sigma_y) & (sub < 0.04 + sigma_y)).sum()} ({((sub > 0.04 - sigma_y) & (sub < 0.04 + sigma_y)).mean() * 100:.1f}%)")
w(f"落在 ±2σ 区间内: {((sub > 0.04 - 2 * sigma_y) & (sub < 0.04 + 2 * sigma_y)).sum()} ({((sub > 0.04 - 2 * sigma_y) & (sub < 0.04 + 2 * sigma_y)).mean() * 100:.1f}%)")
# 每个观测值的翻转变异性：σ噪声下 P(翻转)
from scipy.stats import norm
flip = np.mean([2 * norm.cdf(-abs(y - 0.04) / sigma_y) for y in sub])
w(f"在 σ={sigma_y:.5f} 的独立噪声下, 每行“达标/未达标”状态的平均翻转概率 = {flip * 100:.1f}%")
# 边界样本
w(f"Y浓度在 3.5%-4.5% 之间的行: {((sub >= 0.035) & (sub <= 0.045)).sum()}; 其中按测量值判定达标的: {((sub >= 0.035) & (sub <= 0.045) & (sub >= 0.04)).sum()}")
# Z值 组内SD (男胎)
w("男胎 同次采血重复检测 Z 值组内SD: Z13=0.79, Z18=0.83, Z21=0.89, X=0.60 (均值) —— 均接近或超过1, 说明单次Z值固有噪声大")
# 相邻抽血的真实周进展 vs 噪声
w("\n相邻抽血(同孕妇)的Y浓度变化率(每孕周):")
diffs = []
for pid, g in male.groupby("孕妇代码"):
    g = g.sort_values("_ga")
    g = g.dropna(subset=["_ga", "Y染色体浓度"])
    for i in range(1, len(g)):
        dt = g["_ga"].iloc[i] - g["_ga"].iloc[i - 1]
        if dt > 0.5:
            diffs.append((g["Y染色体浓度"].iloc[i] - g["Y染色体浓度"].iloc[i - 1]) / dt)
diffs = np.array(diffs)
w(f"  相邻检测间隔>0.5周的对数: {len(diffs)}, 中位变化率 = {np.median(diffs):.5f} /周, IQR = [{np.percentile(diffs, 25):.5f}, {np.percentile(diffs, 75):.5f}]")

w("")
w("=" * 90)
w("O. AB x AE 关联检验 (男胎, 行级)")
w("=" * 90)
ct = pd.crosstab(male["_anom"] != "正常", male["胎儿是否健康"])
ct.index = ["AB正常", "AB异常"]
ct.columns = ["AE是", "AE否"]
w(ct.to_string())
chi2, p, dof, _ = stats.chi2_contingency(ct)
n = ct.values.sum()
phi = np.sqrt(chi2 / n)
w(f"chi2={chi2:.3f}, p={p:.4f}, Cramér V={phi:.3f}")
w("结论: AB 标记与出生结局(AE)关联极弱 -> AB 是检测系统的判定结果, 不代表真实健康状态; 且女胎 AE 全为'是', 无法作为验证标签")

w("")
w("=" * 90)
w("P. 有'未达标记录'的孕妇子集: 最早达标周 (左删失较轻的子集)")
w("=" * 90)
never_ok_women = []
first_ok_fail = []
for pid, g in male.groupby("孕妇代码"):
    g = g.dropna(subset=["Y染色体浓度"])
    ok = g[g["Y染色体浓度"] >= 0.04]
    bad = g[g["Y染色体浓度"] < 0.04]
    if len(bad) > 0:
        if len(ok) > 0:
            first_ok_fail.append((pid, ok["_ga"].min(), bad["_ga"].max()))
        else:
            never_ok_women.append(pid)
w(f"曾出现 <4% 记录的孕妇: {len(first_ok_fail) + len(never_ok_women)} (其中后来达标 {len(first_ok_fail)}, 始终未达标 {len(never_ok_women)})")
tf = pd.DataFrame(first_ok_fail, columns=["pid", "first_ok", "last_bad"]).set_index("pid")
bmi1 = male.groupby("孕妇代码").first()["孕妇BMI"]
tf = tf.join(bmi1)
tf["_bmig"] = pd.cut(tf["孕妇BMI"], bins=bins, labels=labels, right=False)
w("子集内 最早达标周 按BMI组:")
w(tf.groupby("_bmig", observed=True)["first_ok"].describe().round(2).to_string())
w("子集内 最晚未达标周 按BMI组:")
w(tf.groupby("_bmig", observed=True)["last_bad"].describe().round(2).to_string())
# 全样本基线（首测即达标的占比极高 → 左删失）
first_rows = male.groupby("孕妇代码").first()
w(f"\n全体孕妇首测孕周: 中位={first_rows['_ga'].median():.2f}w, p25={first_rows['_ga'].quantile(0.25):.2f}w, p75={first_rows['_ga'].quantile(0.75):.2f}w")
w(f"首测即达标比例: {(first_rows['Y染色体浓度'] >= 0.04).mean() * 100:.1f}%")

with open(OUT / "model_analysis.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(R))
print("\nsaved ->", OUT / "model_analysis.txt")
