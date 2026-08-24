# -*- coding: utf-8 -*-
"""EDA 图表生成 -> output/figures/*.png"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

XLSX = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\C题\附件.xlsx")
FIG = Path(r"D:\Users\TtT20\source\repos\数学建模\国赛\25年c题\EDA\output\figures")
FIG.mkdir(parents=True, exist_ok=True)

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
for df in (male, female):
    df["_bmig"] = pd.cut(df["孕妇BMI"], bins=bins, labels=labels, right=False)

C = {"[20,28)": "#8c564b", "[28,32)": "#1f77b4", "[32,36)": "#2ca02c", "[36,40)": "#ff7f0e", "40+": "#d62728"}

# ---------- Fig 1 关键变量分布 ----------
fig, axes = plt.subplots(3, 4, figsize=(15, 9))
panels = [
    ("_ga", "检测孕周(周)", "hist", male),
    ("孕妇BMI", "BMI", "hist", male),
    ("年龄", "年龄(岁)", "hist", male),
    ("体重", "体重(kg)", "hist", male),
    ("Y染色体浓度", "Y染色体浓度", "hist", male),
    ("X染色体浓度", "X染色体浓度", "hist", male),
    ("GC含量", "全局GC含量", "hist", male),
    ("13号染色体的Z值", "13号染色体Z值", "hist", male),
    ("18号染色体的Z值", "18号染色体Z值", "hist", male),
    ("21号染色体的Z值", "21号染色体Z值", "hist", male),
    ("原始读段数", "原始读段数(百万)", "hist", male),
    ("Y染色体浓度", "Y染色体浓度(女胎无此列, 显示X浓度)", "hist", female),
]
for ax, (col, ttl, kind, df) in zip(axes.flat, panels):
    v = pd.to_numeric(df[col], errors="coerce")
    sns.histplot(v.dropna(), bins=40, ax=ax, color="#4c72b0", edgecolor="none")
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8)
fig.suptitle("图1 男胎数据关键变量分布 (女胎最后一格为X浓度)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG / "fig1_distributions.png", bbox_inches="tight")
plt.close(fig)

# ---------- Fig 2 Y浓度 ~ 孕周, 按BMI组 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for lab in labels:
    s = male[male["_bmig"] == lab].dropna(subset=["Y染色体浓度", "_ga"])
    if len(s) == 0:
        continue
    ax.scatter(s["_ga"], s["Y染色体浓度"], s=9, alpha=0.45, c=C[lab], label=f"{lab} (n={len(s)})", edgecolors="none")
ax.axhline(0.04, color="k", ls="--", lw=1.2)
ax.text(28.6, 0.0415, "4% 达标阈值", fontsize=9)
ax.set_xlabel("检测孕周 (周)")
ax.set_ylabel("Y染色体浓度")
ax.set_title("男胎: Y染色体浓度 ~ 孕周 × BMI组 (行级, n=1082)")
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax = axes[1]
# 分组经验中位数曲线 + 分箱达标率
for lab in labels:
    s = male[male["_bmig"] == lab].dropna(subset=["Y染色体浓度", "_ga"])
    if len(s) < 10:
        continue
    s = s.sort_values("_ga")
    ax.plot(s["_ga"], s["Y染色体浓度"].rolling(40, min_periods=10).median(),
            c=C[lab], lw=1.8, label=lab)
ax.axhline(0.04, color="k", ls="--", lw=1.2)
ax.set_ylim(0, 0.18)
ax.set_xlabel("检测孕周 (周)")
ax.set_ylabel("Y染色体浓度 (滚动中位数)")
ax.set_title("按BMI组的Y浓度滚动中位数曲线")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_yconc_vs_ga.png", bbox_inches="tight")
plt.close(fig)

# ---------- Fig 3 达标率(>=4%) ~ 孕周: 经验 + logit ----------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
male["_ok"] = (male["Y染色体浓度"] >= 0.04).astype(int)
for lab in labels:
    s = male[male["_bmig"] == lab].dropna(subset=["_ga", "_ok"])
    if len(s) < 10:
        continue
    # 经验点
    s2 = s.copy()
    s2["_bin"] = np.floor(s2["_ga"])
    emp = s2.groupby("_bin")["_ok"].agg(["mean", "size"])
    emp = emp[emp["size"] >= 10]
    ax.plot(emp.index + 0.5, emp["mean"] * 100, "o", ms=4, c=C[lab], alpha=0.6)
    # logit 拟合
    X = sm.add_constant(s["_ga"])
    m = sm.GLM(s["_ok"], X, family=sm.families.Binomial()).fit()
    g = np.linspace(11, 29, 200)
    p = 1 / (1 + np.exp(-(m.params["const"] + m.params["_ga"] * g)))
    ax.plot(g, p * 100, c=C[lab], lw=2, label=f"{lab} (n={len(s)})")
ax.axhline(80, color="grey", ls=":", lw=1)
ax.text(28.7, 81, "80%", fontsize=8, color="grey")
ax.axhline(90, color="grey", ls=":", lw=1)
ax.text(28.7, 91, "90%", fontsize=8, color="grey")
ax.set_xlabel("检测孕周 (周)")
ax.set_ylabel("Y染色体浓度 ≥ 4% 的比例 (%)")
ax.set_title("图3 男胎: 各BMI组达标率 ~ 孕周 (经验点 + logit 拟合)")
ax.legend(fontsize=8)
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(FIG / "fig3_pass_rate.png", bbox_inches="tight")
plt.close(fig)

# ---------- Fig 4 女胎: Z 值 vs AB 标签 ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, (zc, ttl) in zip(axes, [("21号染色体的Z值", "21号染色体 Z值"),
                                 ("18号染色体的Z值", "18号染色体 Z值"),
                                 ("13号染色体的Z值", "13号染色体 Z值")]):
    order = ["正常", "T21", "T18", "T13"]
    sub = female[female["_anom"].isin(order)]
    sns.boxplot(data=sub, x="_anom", y=zc, order=order, ax=ax,
                palette=["#bbbbbb", "#d62728", "#ff7f0e", "#9467bd"], fliersize=1.5, width=0.6)
    ax.axhline(3, color="k", ls="--", lw=1)
    ax.set_xlabel("AB 非整倍体标签")
    ax.set_ylabel(ttl)
    ax.set_title(f"{ttl}\n(标签与Z值几乎无对应)", fontsize=10)
    ax.tick_params(labelsize=9)
fig.suptitle("图4 女胎: AB 标签对应染色体的 Z 值分布 (虚线=经典阈值3)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / "fig4_female_z_vs_label.png", bbox_inches="tight")
plt.close(fig)

# ---------- Fig 5 同次采血重复检测噪声 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
# 男胎 Y浓度 / Z值 组内SD
sds = {"Y染色体浓度": [], "13号染色体Z值": [], "18号染色体Z值": [], "21号染色体Z值": [], "X染色体Z值": []}
for (pid, draw), g in male.groupby(["孕妇代码", "检测抽血次数"]):
    if len(g) >= 2:
        sds["Y染色体浓度"].append(g["Y染色体浓度"].std())
        sds["13号染色体Z值"].append(g["13号染色体的Z值"].std())
        sds["18号染色体Z值"].append(g["18号染色体的Z值"].std())
        sds["21号染色体Z值"].append(g["21号染色体的Z值"].std())
        sds["X染色体Z值"].append(g["X染色体的Z值"].std())
ax = axes[0]
data = [np.array(v) for v in sds.values()]
bp = ax.boxplot(data, labels=list(sds.keys()), showmeans=True)
ax.set_ylabel("同次采血多次检测的组内SD")
ax.set_title("男胎: 测量噪声(组内SD, 40组重复)\nY浓度SD≈0.54pp ~ 相当于1.3周的浓度进展")
ax.tick_params(labelsize=8, rotation=15)
ax = axes[1]
# 达标率的经验分布 by BMI组 (首测)
first = male.groupby("孕妇代码").first()
gfirst = pd.cut(first["孕妇BMI"], bins=bins, labels=labels, right=False)
okrate = first.groupby(gfirst, observed=True)["Y染色体浓度"].apply(lambda s: (s >= 0.04).mean() * 100)
ax.bar(okrate.index.astype(str), okrate.values, color=[C[l] for l in okrate.index])
ax.set_ylabel("首测即达标比例 (%)")
ax.set_xlabel("BMI 组")
ax.set_title("男胎: 各BMI组孕妇首次检测即达标的比例")
for i, v in enumerate(okrate.values):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(FIG / "fig5_noise_and_firstpass.png", bbox_inches="tight")
plt.close(fig)

# ---------- Fig 6 X浓度: 性别判别 + 女胎X Z值极端行; 人群特征 ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
ax = axes[0]
sns.histplot(male["X染色体浓度"], bins=40, ax=ax, color="#1f77b4", alpha=0.55, label="男胎 (n=1082)")
sns.histplot(female["X染色体浓度"], bins=40, ax=ax, color="#d62728", alpha=0.45, label="女胎 (n=605)")
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_title("X染色体浓度: 男/女胎分布 (男胎均值+5.7%, 女胎-0.7%)")
ax.legend(fontsize=8)
ax.set_xlabel("X染色体浓度")
ax = axes[1]
sub = female[female["X染色体的Z值"].abs() > 3]
sns.scatterplot(data=female, x="_ga", y="X染色体的Z值", s=8, alpha=0.35, ax=ax, color="#2ca02c")
ax.scatter(sub["_ga"], sub["X染色体的Z值"], s=30, color="r", label=f"|X-Z|>3 (n={len(sub)})")
ax.axhline(3, color="k", ls="--", lw=0.8)
ax.axhline(-3, color="k", ls="--", lw=0.8)
ax.set_title("女胎: X染色体Z值 ~ 孕周 (红点=|Z|>3)")
ax.set_xlabel("检测孕周(周)")
ax.set_ylabel("X染色体Z值")
ax.legend(fontsize=8)
ax = axes[2]
sns.boxplot(data=male.assign(表="男胎"), y="孕妇BMI", ax=ax, color="#4c72b0", width=0.5, showfliers=False)
sns.boxplot(data=female.assign(表="女胎"), y="孕妇BMI", ax=ax, color="#dd8452", width=0.5, showfliers=False)
ax.set_title("BMI 分布(孕妇级首批)")
ax.set_ylabel("孕妇BMI")
ax.set_xlabel("")
fig.tight_layout()
fig.savefig(FIG / "fig6_xconc_bmi.png", bbox_inches="tight")
plt.close(fig)

# ---------- Fig 7 女胎: 正常/异常组特征对比热图(均值差异) ----------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
feat = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "X染色体浓度",
        "GC含量", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
        "被过滤掉读段数的比例", "在参考基因组上比对的比例", "重复读段的比例", "孕妇BMI", "年龄", "_ga"]
gmean = female.groupby("_anom")[feat].mean()
gmean = gmean.reindex(["正常", "T21", "T18", "T13", "T13T18"])
zcols = [c for c in feat if "Z值" in c or "浓度" in c]
qcols = [c for c in feat if c not in zcols]
for ax, cols, ttl in zip(axes, [zcols + [_ := "GC含量"], qcols[:8]], ["信号类特征 (均值)", "背景/质量/人群特征 (均值)"]):
    g = gmean[cols].T
    sns.heatmap(g, ax=ax, cmap="RdBu_r", center=None, annot=True, fmt=".3f",
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax.set_title(f"图7 女胎各AB标签 {ttl}")
    ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig7_female_label_heatmap.png", bbox_inches="tight")
plt.close(fig)

print("figures saved:", sorted(p.name for p in FIG.glob("*.png")))
