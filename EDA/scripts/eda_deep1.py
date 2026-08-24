# -*- coding: utf-8 -*-
"""
EDA 深度分析（步骤2）：
A. AB(非整倍体) 与 AE(是否健康) 交叉（行级/孕妇级）
B. 检测抽血次数与多次检测结构（一次采血多次检测）
C. BMI 分组计数（题目建议分组）
D. Y 浓度达标(>=4%) 结构：总体/按孕周/按BMI组；每孕妇最早达标周
E. 相关性 + OLS 回归（问题1 预分析）
F. 女胎按 AB 标签的特征对比（问题4 预分析）
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from scipy import stats

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
male["_anom"] = male["染色体的非整倍体"].fillna("正常")

w("=" * 90)
w("A. 男胎 AB(非整倍体) x AE(是否健康) 交叉（显式核对）")
w("=" * 90)
w("AE 取值: " + {str(k): int(v) for k, v in male["胎儿是否健康"].value_counts().items()}.__str__())
ct = pd.crosstab(male["_anom"], male["胎儿是否健康"])
ct = ct.reindex(columns=["是", "否"]).fillna(0).astype(int)
w(ct.to_string())
w("核对: AB异常行中 AE=否 的数量 = " + str(int(ct.loc[ct.index != "正常", "否"].sum())) + " (应为4)")
w("核对: AB正常行中 AE=否 的数量 = " + str(int(ct.loc["正常", "否"])) + " (应为34)")
# AB=异常的孕妇是否AE=否
ab_m = male[male["_anom"] != "正常"]
w(f"AB 标记异常的行: {len(ab_m)}; 其中 AE=否: {(ab_m['胎儿是否健康'] == '否').sum()}; AE=是: {(ab_m['胎儿是否健康'] == '是').sum()}")
ab_ok = male[male["_anom"] == "正常"]
w(f"AB 标记正常的行: {len(ab_ok)}; 其中 AE=否: {(ab_ok['胎儿是否健康'] == '否').sum()}")
# 按AB类别的AE分布
w("\nAB类别 x AE分布(行级, 归一化):")
w(pd.crosstab(male["_anom"], male["胎儿是否健康"], normalize="index").round(3).reindex(columns=["是", "否"]).to_string())

w("\n孕妇级（每孕妇取任意行判定）: 每孕妇的AB/AE是否恒定")
for col in ["_anom", "胎儿是否健康"]:
    n_uni = male.groupby("孕妇代码")[col].nunique()
    w(f"  {col}: 不变孕妇 {(n_uni == 1).sum()}/{len(n_uni)}, 变化孕妇 {(n_uni > 1).sum()}")
w("AB 在孕妇内变化的组合（示例统计, 只给类别对计数）:")
trans = []
for pid, g in male.groupby("孕妇代码"):
    seq = g.sort_values(["检测抽血次数", "_ga"])["_anom"].tolist()
    uniq = list(dict.fromkeys(seq))
    trans.append(tuple(uniq))
from collections import Counter
cc = Counter(trans)
w("  变化孕妇共 %d 个; 类别序列前10: %s" % (sum(1 for t in trans if len(t) > 1), cc.most_common(10)))
w("  恒定孕妇中各类别数: " + str({k: v for k, v in Counter(t for t in trans if len(t) == 1).most_common()}))
# 同采血次数内多次检测的AB一致性
w("同(孕妇,抽血次数)多次检测的 AB 是否一致:")
cons = 0; total = 0
for (pid, draw), g in male.groupby(["孕妇代码", "检测抽血次数"]):
    if len(g) > 1:
        total += 1
        if g["_anom"].nunique() == 1:
            cons += 1
w(f"  组合数={total}, AB完全一致={cons} ({cons / total * 100:.1f}%)")

w("")
w("=" * 90)
w("B. 检测抽血次数 与 同次采血多次检测")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    grp = df.groupby(["孕妇代码", "检测抽血次数"]).size()
    w(f"[{tag}] (孕妇, 抽血次数) 组合数 = {len(grp)}; 同组合多行（一次采血多次检测）的组合数 = {(grp > 1).sum()}")
    w(f"[{tag}] 组合内行数分布: {grp.value_counts().sort_index().to_dict()}")
    # 同一孕妇多次抽血的间隔（按抽血次数序号 vs 孕周差）
    w(f"[{tag}] 相邻抽血次数的孕周间隔: 待后续分析")

w("")
w("=" * 90)
w("C. BMI 分组（题目建议分组）")
w("=" * 90)
for tag, df in (("男胎", male), ("女胎", female)):
    b = df["孕妇BMI"]
    bins = [20, 28, 32, 36, 40, np.inf]
    labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "40+"]
    g = pd.cut(b, bins=bins, labels=labels, right=False)
    w(f"[{tag}] BMI 分组（行级）: {g.value_counts().sort_index().to_dict()}")
    # 孕妇级（用首行）
    first = df.groupby("孕妇代码").first()
    gfirst = pd.cut(first["孕妇BMI"], bins=bins, labels=labels, right=False)
    w(f"[{tag}] BMI 分组（孕妇级, 首批检测）: {gfirst.value_counts().sort_index().to_dict()}")
    w(f"[{tag}] 孕妇级 BMI 概要: mean={first['孕妇BMI'].mean():.2f} med={first['孕妇BMI'].median():.2f} min={first['孕妇BMI'].min():.2f} max={first['孕妇BMI'].max():.2f}")
    w(f"[{tag}] BMI 缺失: {b.isna().sum()}; BMI<20: {(b < 20).sum()}; BMI>=28 比例: {(b >= 28).mean() * 100:.1f}%")

w("")
w("=" * 90)
w("D. 男胎 Y 浓度达标(>=4%) 结构")
w("=" * 90)
y = male
w(f"Y浓度 >= 4% 行: {(y['Y染色体浓度'] >= 0.04).sum()} / {len(y)} = {(y['Y染色体浓度'] >= 0.04).mean() * 100:.1f}%")
w(f"Y浓度 < 4% 行: {(y['Y染色体浓度'] < 0.04).sum()}")
w(f"Y浓度 < 4% 的孕妇数: {y[y['Y染色体浓度'] < 0.04].groupby('孕妇代码').size().shape[0]}")
# 按孕周分箱
y["_gabin"] = pd.cut(y["_ga"], bins=np.arange(10, 31, 1), right=False)
ga_tab = y.groupby("_gabin", observed=True).agg(
    n=("Y染色体浓度", "size"),
    rate4=("Y染色体浓度", lambda s: (s >= 0.04).mean()),
    mean_conc=("Y染色体浓度", "mean"),
    med_conc=("Y染色体浓度", "median"),
)
w("\n按孕周分箱的达标率/浓度:")
w(ga_tab.round(3).to_string())
# 按 BMI 组 x 孕周分箱
w("\n按 BMI 组 x 孕周分箱 (达标率%, 行数):")
bins = [20, 28, 32, 36, 40, np.inf]
labels = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "40+"]
y["_bmig"] = pd.cut(y["孕妇BMI"], bins=bins, labels=labels, right=False)
piv = y.pivot_table(index="_bmig", columns="_gabin", values="Y染色体浓度",
                    aggfunc=lambda s: (s >= 0.04).mean() * 100, observed=True)
w(piv.round(1).to_string())
w("\n行数:")
w(y.pivot_table(index="_bmig", columns="_gabin", values="Y染色体浓度", aggfunc="size", observed=True).to_string())
# 每孕妇最早达标孕周（第一个 Y浓度>=4% 的行）
first_ok = {}
for pid, g in y.dropna(subset=["Y染色体浓度"]).groupby("孕妇代码"):
    g = g[g["Y染色体浓度"] >= 0.04]
    first_ok[pid] = g["_ga"].min() if len(g) else np.nan
fo = pd.Series(first_ok)
w(f"\n每孕妇最早达标孕周(>=4%): 可估算孕妇 {fo.notna().sum()}/{len(fo)} (即从未达标的孕妇 {fo.isna().sum()})")
w(f"  mean={fo.mean():.2f}, 中位={fo.median():.2f}, min={fo.min():.2f}, max={fo.max():.2f}, p25={fo.quantile(0.25):.2f}, p75={fo.quantile(0.75):.2f}")
# 首行即达标的比例
first_rows = y.groupby("孕妇代码").first()
w(f"首次检测即达标(Y>=4%)的孕妇: {(first_rows['Y染色体浓度'] >= 0.04).sum()}/{len(first_rows)} = {(first_rows['Y染色体浓度'] >= 0.04).mean() * 100:.1f}%")
# 按BMI组的孕妇最早达标周
w("\n按BMI组：孕妇最早达标孕周分布（含从未达标）")
first_bmi = male.groupby("孕妇代码").first()[["孕妇BMI"]]
fo2 = fo.rename("firstwk").to_frame().join(first_bmi)
fo2["_bmig"] = pd.cut(fo2["孕妇BMI"], bins=bins, labels=labels, right=False)
res = []
for lab in labels:
    sub = fo2[fo2["_bmig"] == lab]["firstwk"]
    res.append({"BMI组": lab, "n孕妇": int(len(sub)), "从未达标": int(sub.isna().sum()),
                "med_firstwk": round(float(sub.median()), 2), "mean_firstwk": round(float(sub.mean()), 2),
                "p25": round(float(sub.quantile(0.25)), 2), "p75": round(float(sub.quantile(0.75)), 2),
                "min": round(float(sub.min()), 2), "max": round(float(sub.max()), 2)})
w(pd.DataFrame(res).to_string())
# 从未达标的孕妇BMI分布
never = fo2[fo2["firstwk"].isna()]
w(f"\n从未达标的孕妇 ({len(never)}人) BMI: mean={never['孕妇BMI'].mean():.2f}, med={never['孕妇BMI'].median():.2f}, min={never['孕妇BMI'].min():.2f}, max={never['孕妇BMI'].max():.2f}")

w("")
w("=" * 90)
w("E. 相关性（问题1 预分析, 男胎, 行级）")
w("=" * 90)
cols = ["Y染色体浓度", "_ga", "孕妇BMI", "年龄", "身高", "体重", "X染色体浓度", "Y染色体的Z值",
        "原始读段数", "唯一比对的读段数", "GC含量", "被过滤掉读段数的比例", "重复读段的比例"]
sub = male[cols].dropna()
w(f"样本量(行, 完全无缺失) = {len(sub)}")
pear = sub.corr(method="pearson")["Y染色体浓度"].round(3)
spea = sub.corr(method="spearman")["Y染色体浓度"].round(3)
w("与 Y染色体浓度 的 Pearson / Spearman 相关:")
for c in cols[1:]:
    r_p, p_p = stats.pearsonr(sub[c], sub["Y染色体浓度"])
    r_s, p_s = stats.spearmanr(sub[c], sub["Y染色体浓度"])
    w(f"  {c}: r={r_p:.3f}(p={p_p:.2e}) | rho={r_s:.3f}(p={p_s:.2e})")
# 相关性矩阵（孕周/BMI/年龄/身高/体重）
w("\n主要字段Pearson相关矩阵(n={}):".format(len(sub)))
w(sub[["_ga", "孕妇BMI", "年龄", "身高", "体重", "X染色体浓度"]].corr().round(3).to_string())
# 孕周与BMI分组的Y浓度均值曲线（用于问题2/3 建模参考）
w("\n各BMI组内 Y浓度~孕周 线性拟合 (ols, 无聚类):")
try:
    import statsmodels.api as sm
    for lab in labels:
        s = y[(pd.cut(y["孕妇BMI"], bins=bins, labels=labels, right=False) == lab)].dropna(subset=["Y染色体浓度", "_ga"])
        if len(s) < 10:
            continue
        X = sm.add_constant(s["_ga"])
        m = sm.OLS(s["Y染色体浓度"], X).fit()
        w(f"  {lab}: n={len(s)} 截距={m.params['const']:.4f} 斜率={m.params['_ga']:.5f} R2={m.rsquared:.3f} p(斜率)={m.pvalues['_ga']:.2e}")
        # 达到4%的孕周 = (0.04-截距)/斜率
        t = (0.04 - m.params["const"]) / m.params["_ga"]
        w(f"      达到4%孕周(线性回归估计): {t:.2f} 周")
except Exception as e:
    w("statsmodels 不可用:", e)

w("")
w("=" * 90)
w("F. 女胎按 AB 标签的特征对比（问题4 预分析）")
w("=" * 90)
f = female
f["_anom"] = f["染色体的非整倍体"].fillna("正常")
w("AB 类别计数(行级): " + f["_anom"].value_counts().to_dict().__str__())
w("AB 类别计数(孕妇级): " + f.groupby("孕妇代码").first()["_anom"].value_counts().to_dict().__str__())
# 每孕妇 AB 是否恒定
n_uni = f.groupby("孕妇代码")["_anom"].nunique()
w(f"孕妇级 AB 恒定: {(n_uni == 1).sum()}/{len(n_uni)}; 变化: {(n_uni > 1).sum()}")
# 特征均值对比
feat = ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值",
        "X染色体浓度", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
        "GC含量", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例",
        "唯一比对的读段数", "被过滤掉读段数的比例", "孕妇BMI", "年龄", "_ga",
        "身高", "体重", "生产次数"]
# 怀孕次数为混合类型(1/2/≥3)，单独看
f["_pregency_cat"] = pd.to_numeric(f["怀孕次数"], errors="coerce").fillna(3).astype(int)  # ≥3 记 3
feat.append("_pregency_cat")
w("\n各 AB 类别特征均值 (n行):")
grouped = f.groupby("_anom")[feat].agg(["mean", "median"])
w(grouped.round(3).to_string())
# 正常 vs 异常 的 Z 区分度（AUC 简评）
from sklearn.metrics import roc_auc_score
f["_abn"] = (f["_anom"] != "正常").astype(int)
w("\n正常(0) vs 异常(1) 各 Z 值的 AUC:")
for c in ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值"]:
    try:
        auc = roc_auc_score(f["_abn"], f[c])
        w(f"  {c}: AUC={auc:.3f}")
    except Exception as e:
        w(f"  {c}: {e}")
# T21 vs 正常
for target in ["T21", "T18", "T13"]:
    sub2 = f[f["_anom"].isin(["正常", target])]
    y2 = (sub2["_anom"] == target).astype(int)
    aucs = {c: roc_auc_score(y2, sub2[c]) for c in ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]}
    w(f"  {target} vs 正常 (n={len(sub2)}): { {k: round(v, 3) for k, v in aucs.items()} }")
# 女胎 X 浓度负值比例
w(f"\n女胎 X染色体浓度 < 0 比例: {(f['X染色体浓度'] < 0).mean() * 100:.1f}%; 男胎 X浓度 < 0 比例: {(male['X染色体浓度'] < 0).mean() * 100:.1f}%")
# 女胎 Z>3 比例
for c in ["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]:
    w(f"女胎 {c} > 3 行: {(f[c] > 3).sum()}, > 5: {(f[c] > 5).sum()}; 男胎 {c} > 3: {(male[c] > 3).sum()}, >5: {(male[c] > 5).sum()}")

with open(OUT / "deep_analysis.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(R))
print("\nsaved ->", OUT / "deep_analysis.txt")
