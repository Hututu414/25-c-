# Q4 最终 Soft Fusion 摘要

## Fusion 排名

| Rank | Fusion | PR-AUC | ROC-AUC | Brier | Calibration slope |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | Maximum risk | 0.548943 | 0.805576 | 0.072856 | 0.827427 |
| 2 | Noisy-OR | 0.545855 | 0.804004 | 0.073792 | 0.858420 |
| 3 | Logistic stacking | 0.539826 | 0.772791 | 0.076926 | 1.564355 |

PR-AUC 差距小于 0.01 时按预设规则优先选择更简单、可解释的融合，因此最佳 fusion = Maximum risk。

## Youden 判定

| Rule | Recall | Specificity | Precision | F1 | F2 | Balanced Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old: chromosome Minimax to hard OR | 0.867589 | 0.290057 | 0.136734 | 0.236236 | 0.419324 | 0.578823 |
| Direct Any M1-U | 0.906126 | 0.220720 | 0.130970 | 0.228860 | 0.414946 | 0.563423 |
| Maximum risk + Youden | 0.591897 | 0.836358 | 0.319175 | 0.414717 | 0.505510 | 0.714128 |
| Noisy-OR + Youden | 0.666008 | 0.767711 | 0.270931 | 0.385174 | 0.515628 | 0.716859 |
| Logistic stacking + Youden | 0.624506 | 0.841353 | 0.337839 | 0.438476 | 0.533900 | 0.732930 |

## 高敏感策略

最佳 fusion（Maximum risk）在 Recall>=90% 约束下：Recall=0.872530，Specificity=0.424985，Precision=0.164349；所列阈值为五个 training-only outer-fold 阈值的中位数 0.024053。

## 最终推荐

最佳 fusion = Maximum risk

最终阈值规则 = Recall>=0.90（逐 outer fold 仅由 training OOF 分数选择）

是否显著改善旧 OR 的 Specificity = 是（0.290057 -> 0.424985）

是否显著改善 Precision = 是（0.136734 -> 0.164349）

Recall 下降多少 = -0.004941（0.867589 -> 0.872530；负值表示反而上升）

是否建议冻结 Q4 = 是

