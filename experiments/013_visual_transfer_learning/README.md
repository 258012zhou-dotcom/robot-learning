# 实验 013：视觉预训练与小样本微调

## 目标

在未旋转形状图像上预训练视觉 Encoder，再用每轮仅 40 张旋转目标图像适配，比较从零训练、冻结 Linear Probe 和完整微调。

## 对照方法

- `scratch`：Encoder 与 Head 都随机初始化并在目标小数据上训练。
- `linear_probe`：加载预训练 Encoder、完全冻结，只训练新分类 Head。
- `full_finetune`：加载同一预训练 Encoder，Encoder 用较小学习率，Head 用较大学习率。

目标训练重复 3 次，每次使用不同的 40 张标注图；验证集随试验改变，测试集保持固定。结果报告平均值和总体标准差。

## 额外检查

- 预训练模型在来源测试集上的准确率。
- 未适配时在旋转目标测试集上的 Zero-shot 准确率。
- 适配后把原始 Source Head 接到适配后的 Encoder 上，再评价来源测试集，隔离 Encoder 的潜在遗忘。
- 另行保存“目标 Head 直接评价来源数据”的跨域准确率，但不把它解释为 Encoder 遗忘。
- 单元测试确认 Linear Probe 的 Encoder 权重不变，完整微调的 Encoder 确实更新。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_transfer_learning.py
PYTHONPATH=src python experiments/013_visual_transfer_learning/run.py
python -m json.tool outputs/013_visual_transfer_learning/results.json
```

## 输出

- `outputs/013_visual_transfer_learning/results.json`
- `outputs/013_visual_transfer_learning/accuracy_comparison.png`
- `outputs/013_visual_transfer_learning/first_trial_learning_curves.png`
- `outputs/013_visual_transfer_learning/run.log`

## 实际结果

使用 RTX 5060 Laptop GPU、固定预训练模型和三组目标小样本运行：

- 来源域预训练测试准确率为 `100.00%`。
- 未适配的旋转目标域 Zero-shot 准确率为 `50.00%`。
- Scratch 目标准确率为 `66.83% ± 1.74%`。
- Frozen Linear Probe 目标准确率为 `56.75% ± 9.37%`。
- Full Fine-tuning 目标准确率为 `86.50% ± 1.14%`。

接回原始 Source Head 后：

- Linear Probe 的来源准确率为 `100.00% ± 0.00%`，符合 Encoder 完全冻结的预期。
- Full Fine-tuning 的来源准确率为 `66.11% ± 8.23%`，相对预训练基线平均下降约 33.89 个百分点。

因此本实验中的预训练没有直接带来 Zero-shot 或稳定线性可分的旋转特征，但为完整微调提供了明显优于随机初始化的起点。目标域提升同时伴随来源表示兼容性下降。

## 当前边界

来源与目标都是同一合成生成器的两类形状，领域偏移仅以正方形旋转为主。该实验用于学习迁移控制，不代表真实机器人视觉迁移效果。
