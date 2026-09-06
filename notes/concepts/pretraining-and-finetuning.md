# 预训练、微调与迁移评价

## 快速复习

预训练在较大来源数据上学习可复用参数，微调在较小目标数据上适配具体任务：

```text
source data → pretraining → pretrained encoder
target data + pretrained encoder → adaptation → downstream model
```

需要区分：

- Zero-shot：不使用目标训练样本直接评价。
- Linear Probe：冻结 Encoder，只训练线性 Head。
- Partial Fine-tuning：只解冻部分层。
- Full Fine-tuning：更新全部 Encoder 与 Head。
- PEFT：通过 LoRA、Adapter 等少量参数适配。

来源准确率高不代表目标域一定有效；目标性能提升也可能伴随灾难性遗忘。结论必须同时和 Scratch、冻结基线及来源能力比较。

## Pretraining 与 Downstream Task

预训练目标可以与最终任务不同：

- 图像分类或对比学习 → 检测、分割、机器人视觉。
- 文本预测 → 指令理解和对话。
- 图文对齐 → 视觉问答和视觉语言条件任务。
- 多机器人轨迹 → 特定机器人策略。

迁移能否成功取决于来源表示中是否保留目标任务需要的信息，以及目标适配是否有能力重新组织这些信息。

实验 013 中，来源模型在轴对齐形状上达到 100%，旋转目标 Zero-shot 只有 50%。这说明来源任务允许模型依赖方向特征，没有自动学到旋转不变性。

## Checkpoint 的内容

迁移通常至少需要：

```text
model state_dict
model architecture/configuration
input preprocessing
class/token mapping
normalization statistics
```

恢复中断训练还需要：

```text
optimizer state
learning-rate scheduler state
epoch/step
random state
best validation metric
```

用于下游迁移时，通常加载模型参数并重新创建优化器。结构名称和张量形状必须匹配；成功加载权重也不代表输入单位、通道、Tokenizer 或动作归一化正确。

## 替换 Head

预训练模型常写成：

```text
encoder → source head
```

目标任务适配：

```text
pretrained encoder → new target head
```

新 Head 随机初始化。若类别数量或输出语义改变，不能直接沿用旧 Head。实验 013 为目标任务重新初始化二分类 Head，并分别冻结或更新 Encoder。

## Frozen Encoder 与 Linear Probe

冻结：

```python
for parameter in encoder.parameters():
    parameter.requires_grad_(False)
```

优化器应只包含仍可训练的 Head 参数。Linear Probe 主要回答：目标信息是否已经能从冻结表示中被线性读取。

Linear Probe 失败不等于预训练参数完全无用。目标信息可能需要非线性读取，或 Encoder 需要改变才能适应领域偏移。

实验 013 的 Linear Probe 目标准确率为 `56.75% ± 9.37%`，低于 Scratch 且波动大；但完整微调达到 `86.50% ± 1.14%`。这说明预训练表示不是稳定线性可用，却仍是有价值的可调整起点。

## Full Fine-tuning 与分层学习率

完整微调更新 Encoder 和新 Head。常采用：

```text
encoder learning rate < head learning rate
```

原因是 Encoder 已包含有用参数，需要较小更新；Head 从随机初始化开始，需要较快学习。

项目优化器使用独立参数组，并由测试确认两组学习率和冻结状态真实生效。实验 013 使用 Encoder `2e-4`、Head `1e-3`。

完整微调适应能力更强，但小数据下可能过拟合或破坏来源表示。可以考虑先训练 Head、逐步解冻、层级学习率、正则化或 PEFT。

## 灾难性遗忘的正确评价

若适配时替换了 Head，直接使用 Target Head 评价来源数据会混合两个变化：

```text
Encoder 是否改变
Target Head 是否适合 Source 表示
```

要隔离 Encoder 表示漂移，应使用：

```text
adapted encoder + original source head → source test
```

实验 013 初版评价设计混淆了这两点。修正后：

- Linear Probe Encoder 冻结，接回原 Head 后来源准确率严格保持 100%。
- Full Fine-tuning 接回原 Head 后来源准确率为 `66.11% ± 8.23%`。

这才支持“完整微调导致来源表示兼容性下降”的结论。Target Head 在来源数据上的表现可以另外记录，但不应单独称为灾难性遗忘。

更严格的持续学习评价还可以重新训练 Source Linear Probe、评价多个来源任务，或测量参数/表示漂移。

## 公平迁移实验

至少比较：

```text
scratch
pretrained + frozen probe
pretrained + full fine-tuning
```

应控制：

- 相同目标训练、验证和测试数据。
- 相同模型容量或明确报告差异。
- 相同训练预算和评价指标。
- 测试集不用于选择超参数。
- 多个随机种子或数据子集。
- 来源能力与目标能力分别评价。

机器人数据应按轨迹、场景、物体或任务拆分，避免同一视频相邻帧跨越训练集和测试集造成泄漏。

## 实验 013 的证据

使用三组不同的 40 张目标标注图，固定旋转目标测试集：

| 方法 | 目标准确率 | 原 Source Head 下来源准确率 |
|---|---:|---:|
| Scratch | `66.83% ± 1.74%` | `50.56% ± 3.76%`，仅作兼容参考 |
| Linear Probe | `56.75% ± 9.37%` | `100.00% ± 0.00%` |
| Full Fine-tuning | `86.50% ± 1.14%` | `66.11% ± 8.23%` |

Full Fine-tuning 比 Scratch 高约 19.67 个百分点，是当前受控条件下的正迁移证据。与此同时来源兼容性下降，说明迁移存在适应—保持 trade-off。

三次重复只能初步观察波动，不能替代更大样本、更多随机种子和置信区间。

## 工程易错点

检查点至少保存“如何重建模型”和“如何解释输入输出”：架构参数、权重、类别/Token 顺序、预处理、配置及选模依据。实验 008/009 已增加最佳验证模型保存，015 保存基础模型与 LoRA 的组合；运行入口可用 `--checkpoint --image --device` 直接推理，具体命令见各实验 README。

验证方式是换一个进程重新建模、加载，再对同一输入比较预测；只有文件存在不能证明可复用。本次 CPU 加载测试已覆盖全部权重与输出一致性，GPU 路径仍需在 CUDA 可用的执行环境重跑。推理恢复不等于断点续训：后者还需要优化器等训练状态。

- 冻结参数后仍把所有参数交给优化器，虽然通常不会更新，但意图不清晰。
- `model.train()` 会影响 Dropout 和 BatchNorm；冻结梯度不等于冻结运行状态。
- 加载 Encoder 后忘记重新初始化目标 Head。
- 用 Target Head 的来源准确率误判 Encoder 遗忘。
- 组合不同模型部件时，一个在 CPU、另一个在 CUDA。
- 输入预处理、类别索引或归一化与预训练不一致。
- 只报告最好一次目标结果，忽略随机波动和来源能力下降。

项目中的组合评价函数会把新建 Source Head 模型自动移动到适配 Encoder 所在设备，并由测试检查所有参数设备一致。

## 具身智能中的迁移

具身迁移可能跨越：

- 仿真到真实。
- 网络图像到机器人相机。
- 一台机器人到另一台机器人。
- 一种任务到多任务。
- 通用视觉语言模型到视觉语言动作策略。

除模型权重外，还必须对齐相机视角、控制频率、动作坐标系、状态定义、动作归一化和安全边界。参数成功加载不代表 observation/action 语义兼容。

当前项目只验证两类合成图像的旋转领域偏移。真实迁移仍需更复杂数据、Partial/PEFT 对照、来源数据混合和真实下游任务评价。
