# Q4 最终决策层摘要

## Any-abnormal probability

PR-AUC = 0.531123

Brier = 0.075000

Calibration slope = 0.762070

Calibration intercept = -0.388408

## 二分类决策

旧 OR：Recall=0.867589, Specificity=0.290057, Precision=0.136734, F2=0.419324, Balanced Accuracy=0.578823

新 direct Any：Recall=0.906126, Specificity=0.220720, Precision=0.130970, F2=0.414946, Balanced Accuracy=0.563423

## 三级判定（alpha=beta=0.10）

tau_L（outer-fold median） = 0.011745

tau_H（outer-fold median） = 0.227206

Normal比例 = 0.195395

Suspicious比例 = 0.666040

Abnormal比例 = 0.138565

Coverage = 0.333960

Selective Accuracy = 0.732766

NPV = 0.944876

PPV = 0.433665

严格 OOF 使用每个 outer fold 单独由 outer-training OOF 选择的阈值；上面 tau_L/tau_H 仅报告五折中位数。

## 染色体定位

T13 F1 = 0.534435

T18 F1 = 0.615532

T21 F1 = 0.218750（探索性；样本极少）

Exact-match accuracy = 0.179842

## 最终判断

是否冻结 M1-U = 是

是否采用 direct Any-abnormal gate = 否

是否采用 Normal/Suspicious/Abnormal 三级决策 = 否

是否保留 chromosome-specific 第二阶段定位 = 是（T21 仅探索性报告）

Q4 是否可以正式冻结 = 否
