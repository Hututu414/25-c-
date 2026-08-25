# Q4 第二轮定向改进摘要

## 概率改进

M1-raw PR-AUC / Brier = 0.4515 / 0.3417

M1-cal PR-AUC / Brier = 0.4263 / 0.3447

M1-U PR-AUC / Brier = 0.5459 / 0.0738

M1-H PR-AUC / Brier = 0.5239 / 0.0758

M1-HP PR-AUC / Brier = 0.4792 / 0.0844

## 验证协议影响

最终候选 = M1-U

Accuracy 变化（70/30 减 strict）= +0.0795

Weighted F1 变化（70/30 减 strict）= +0.0928

row-level train/test 共同孕妇数 = 107

## 最终建议

第一推荐：M1-U

第二推荐：M1-H

是否保留 Z 校准层：是，M1-U 使用 calibrated Z*，M1-H 作为 hybrid 次选

下一步是否已经可以冻结 Q4：可以；建议经用户确认后冻结第一推荐与第二推荐，不由本脚本自动执行冻结。
