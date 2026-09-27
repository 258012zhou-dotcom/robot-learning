# 实验 047：低维 ACT 关键机制

## 目标与原理

ACT（Action Chunking with Transformers）把一个观察映射成未来一小段动作。训练时，CVAE 的后验编码器还能看到示范动作片段，从而估计潜变量 `z`；推理时没有未来示范动作，改用固定的先验均值 `z=0`。Transformer 解码器输出 `H=4` 个动作。片段末尾补齐的动作使用显式掩码排除，目标函数为有效动作的 L1 加上加权 KL。

这只是**低维状态输入的教学规模机制练习**：没有相机图像、ACT 原论文的完整视觉结构或机器人真机数据，也没有复现论文结果。数据仍来自实验 020 的仿真 P 控制专家；划分和训练集归一化沿用实验 046。

## 如何验证

单元测试检查输出与梯度、补齐掩码、L1/KL 手算、推理时不依赖示范动作。实际运行用 validation 的**先验推理片段 L1**选择模型，最后只对 test 评估一次；同时报告训练后验重建误差和推理先验误差，避免把训练时的额外信息泄漏成执行能力。保存并重新加载 checkpoint 后核对相同输入的推理输出。

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_act_lowdim.py tests/test_action_chunk_dataset.py tests/test_sequence_behavior_cloning.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/047_act_lowdim_mechanics/run.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/047_act_lowdim_mechanics/diagnose_latent.py
```

配置在 `configs/047_act_lowdim_mechanics.json`，输出在 Git 忽略的 `outputs/047_act_lowdim_mechanics/`。运行需要实验 020 的 `dataset.npz` 与 `manifest.json`。

## 实际结果与边界

2026-09-27，在 CPU 上用 seed 42 训练 20 轮；train/validation/test 分别为 14/3/3 条仿真专家 Episode，3276/709/723 个片段起点。模型有 22,737 个参数，依据 validation **先验推理片段 L1** 选出第 18 轮；checkpoint 重新加载后，同一输入的推理输出一致。

| 指标 | 初始化时 validation | 第 18 轮 validation | 最终 test |
| --- | ---: | ---: | ---: |
| 后验重建片段 L1（训练时可见示范动作） | 0.5603 | 0.00358 | 0.00503 |
| 先验推理片段 L1（不见示范动作） | 0.5650 | 0.00287 | 0.00444 |
| 先验推理第一动作 L1 | 0.6557 | 0.00328 | 0.00539 |
| 平均 KL | 1.636 | 0.00101 | 0.00071 |

在这个简单、近乎确定性的仿真专家数据上，KL 接近零，后验重建甚至略差于固定 `z=0` 的先验推理。合理解释是模型几乎没有利用潜变量，后验采样噪声反而增加误差；这不是多模态动作学习成功的证据。低 L1 只说明离线模仿拟合，**没有开展闭环评价**；不能与实验 046 的 MSE 直接比较，也不能声称 ACT 比非 ACT 基线更好。下一步先理解潜变量何时必要，再决定是否在具有多种合理动作的任务上验证，而非继续在此任务调参。

### 固定观察、改变潜变量的补充诊断

为避免只凭 KL 推断“模型忽略了 `z`”，加载同一个 checkpoint，固定测试集第 1 个归一化观察，先核对显式 `z=0` 与默认推理完全一致；然后每次只把一个潜变量坐标改成 `+1` 或 `−1`。8 个坐标里，这个观察上响应最大的第 5 维（代码索引 4）使四步动作中最大绝对变化为 `0.00383`；其余坐标的最大变化均不超过 `0.00205`。沿第 5 维从 `−2` 扫到 `+2` 时，片段第 4 步动作从 `0.69924` 变到 `0.68484`，基准 `z=0` 为 `0.69324`；它们都仍是同方向动作。

这说明**潜变量确实影响了数值输出，但在这个观察上影响很小，没有展示出不同动作模式**。它不证明所有观察都如此，也不能单凭一个观察的干预判断模型在复杂任务上是否会使用 `z`。原始数值见忽略提交的 `outputs/047_act_lowdim_mechanics/latent_intervention.json`；这是离线模型诊断，不是闭环测试。
