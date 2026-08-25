# Q4 第一轮模型筛选摘要

## 校准是否有效

原始 Z>3 的 Any Recall / F2 = 0.0356 / 0.0398

校准 Z*>3 的 Any Recall / F2 = 0.0939 / 0.1111

M1-raw PR-AUC = 0.4515

M1-calibrated PR-AUC = 0.4263

结论：Z 重新校准是否值得保留？作为候选模块值得保留，但尚未证明具有一致判别增益：固定规则的 Recall/F2 提高，校准 M1 的 Recall/F2 与 Balanced Accuracy 提高，但 PR-AUC 变化为 -0.0253。以上均为严格孕妇级外层 OOF、patient-weighted 指标。

## 模型排名

| Rank | Model | Any PR-AUC | Recall | F2 | Balanced Accuracy | Brier |
|---:|---|---:|---:|---:|---:|---:|
| 1 | M1 calibrated Elastic-Net | 0.4263 | 0.9239 | 0.4391 | 0.5993 | 0.3447 |
| 2 | M2 calibrated Random Forest | 0.4115 | 0.8498 | 0.4096 | 0.5669 | 0.1181 |
| 3 | M3 calibrated LightGBM | 0.3608 | 0.9358 | 0.3980 | 0.5259 | 0.1039 |


## 各染色体

T13 最佳模型及 PR-AUC = M2 calibrated Random Forest / 0.3603

T18 最佳模型及 PR-AUC = M1 calibrated Elastic-Net / 0.5423

T21 最佳模型及 PR-AUC = M1 calibrated Elastic-Net / 0.1005

## 标签噪声

低质量样本数量 = 42 条记录（34 位孕妇）

技术重复标签不一致率 = 15.6250%（按具有重复记录的 technical-repeat group 计；按重复记录加权为 14.4578%）

低质量样本上的模型表现是否明显恶化 = 否（未见明显描述性下降）

## 最终建议

第一推荐：M1 calibrated Elastic-Net

第二推荐：M2 calibrated Random Forest

不推荐：M3 calibrated LightGBM（本轮相对前两者无决策优势）

建议继续把“质量门控 → 条件 Z 校准 → 多标签概率分类 → 风险阈值”作为 Q4 候选统一框架，但下一轮必须保留 M1-raw 对照，不能把条件 Z 校准视为已证实的必要环节。本轮仅评估对题目 AB 判定标签的复现能力；AB 不是经核型验证的真值。Q4 尚未冻结，未建立三级判定。
