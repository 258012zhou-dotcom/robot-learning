# 实验反思

## 待验证问题

- 预训练 Encoder 在旋转目标域上的 Zero-shot 能力如何？
- 40 张目标标注下，预训练是否优于从零训练？
- Frozen Linear Probe 是否足够，还是必须更新 Encoder？
- 完整微调提高目标性能时，是否降低来源域性能？
- 三次小样本试验之间的波动有多大？

## 解释原则

- 同时报告三次结果、均值和标准差，不只选择最好一次。
- Linear Probe 失败不等于预训练完全无用，可能是目标信息不易线性读取。
- Full Fine-tuning 提升也要与 Scratch 比较，不能只与 Zero-shot 比较。
- 来源域性能下降用于观察遗忘，但来源与目标并非完全独立任务。

## 实际结果

- Source Pretraining：来源测试 `100.00%`，旋转目标 Zero-shot `50.00%`。
- Scratch：目标测试 `66.83% ± 1.74%`。
- Linear Probe：目标测试 `56.75% ± 9.37%`。
- Full Fine-tuning：目标测试 `86.50% ± 1.14%`。
- 原 Source Head 接回适配 Encoder 后，Linear Probe 来源准确率保持 `100.00%`，Full Fine-tuning 为 `66.11% ± 8.23%`。

## 现象解释

来源模型在轴对齐图像上满分、在旋转目标域只有随机水平，说明高来源准确率不自动代表领域外泛化。预训练任务没有迫使 Encoder 学到足够的旋转不变表示。

Linear Probe 冻结 Encoder 后平均表现低于 Scratch，且三次结果从 `50.00%` 到 `70.00%` 波动较大。这说明少量目标标签只能偶尔从冻结特征中找到有用线性边界，无法稳定补偿来源表示与旋转目标之间的偏移。

Full Fine-tuning 平均比 Scratch 高约 19.67 个百分点，说明预训练参数虽然不能直接解决目标任务，却提供了有用的优化起点。完整更新 Encoder 后，模型可以用 40 张目标图适应旋转变化。

这种适应伴随来源表示变化：使用原 Source Head 评价时，Full Fine-tuning 的来源准确率平均从 100% 降至 66.11%。Linear Probe 保持 100%，证明修正后的指标能够区分“Head 改变”与“Encoder 遗忘”。

## 评价设计修正

初版错误地使用新 Target Head 直接评价来源数据，并把结果称为来源保留率。这个指标混合了 Head 改变和 Encoder 改变，导致冻结 Encoder 的 Linear Probe 也会显示来源准确率下降。

修正后同时保留两个指标：

- `source_accuracy_with_target_head`：目标 Head 在来源域的跨域兼容性。
- `source_accuracy_with_original_source_head`：适配 Encoder 与原 Source Head 的兼容性，用于观察 Encoder 表示漂移。

修正过程中还发现组合模型的 Encoder 在 CUDA、Head 在 CPU，触发矩阵设备不一致错误。组合函数现在自动将整个评价模型移动到适配 Encoder 所在设备，并增加设备一致性测试。

## 限制与下一步

- 每轮只有 40 张目标训练图，三轮重复仍不足以给出精确统计置信区间。
- 来源与目标共享生成器、类别和背景分布，只改变正方形旋转，远比真实领域迁移简单。
- Full Fine-tuning 多数最佳轮次接近训练上限，尚未系统比较训练预算与早停。
- 没有使用混合来源数据、蒸馏或正则化缓解遗忘。
- 没有比较 Partial Fine-tuning 和参数高效微调。

下一步学习视觉语言模型基础，之后再学习参数高效微调和多模态对齐。
