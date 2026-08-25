# Q4 最终冻结工作区

## 最终采用方案

- 主任务：`AB` 为空判为 Normal，非空判为 Abnormal。
- 底层模型：chromosome-specific M1-U，输出严格 outer patient-level OOF 的 `p13`、`p18`、`p21`。
- 最终融合：`Maximum risk = max(p13, p18, p21)`。
- 最终阈值：每个 outer fold 仅使用 training OOF 分数选择的 `Recall>=0.90` 高敏阈值。
- 最终严格 OOF：Recall=0.872530，Specificity=0.424985，Precision=0.164349。
- QC：只保留 `qc_low_confidence` 置信度标记，不改变二分类判定。
- 染色体定位：异常后取 `argmax(p13,p18,p21)`；T21 仅作探索性解释。

## 正式入口与材料

- `code/run_q4_final_fusion.py`：最终 soft-fusion 决策层入口。
- `code/run_q4_round2.py`：生成最终采用的 M1-U 染色体 OOF 概率；文件内其他候选分支属于模型筛选记录。
- `code/run_q4_models.py`、`code/prepare_q4_data.py`、`code/calibrate_z.R`：数据准备、校准及共享建模基础。
- `code/build_workbooks.mjs`：Excel 构建器。
- `data_processed/q4_round2_oof_predictions.csv`：最终 `p13/p18/p21` 权威输入。
- `data_processed/q4_oof_predictions.csv`：QC 与基础 OOF 信息。
- `data_processed/q4_decision_layer_oof_predictions.csv`：仅保留 Direct Any 参考概率，不是最终模型。
- `outputs/final_fusion/`：最终 5 份 Excel、2 张图及摘要。
- `outputs/reference/`：国奖论文多分类指标参考，不作为本轮二分类主评价。

## 尝试过程归档

- `Archive folder/first_round/`：第一轮模型海选、校准诊断、原始比较表和图。
- `Archive folder/second_round/`：第二轮 M1-U/M1-H/M1-HP/RF-paper 等候选比较与论文协议参考。
- `Archive folder/decision_layer_attempt/`：未采用的 Direct Any、三级 Normal/Suspicious/Abnormal 和两阶段定位方案。

归档结果仅用于追溯，不应与 `outputs/final_fusion/` 中的正式结论混用。Q1–Q3 保持冻结，本次未修改；未执行 Git stage、commit 或 push。
