# 实验 044：从单步转移到动作片段

## 目标与原理

实验 020 的每行是 `observation[t] → action[t] → next_observation[t]`。序列策略的一个基本标签可以是：给定当前 `observation[t]`，预测从 `action[t]` 开始的连续 `H` 个动作。本实验取 `H=4`，只验证**时序标签如何形成**，不训练 ACT 或其他序列模型。

`Episode` 最后几步不足 4 个动作时，保留起点样本，将缺少的位置补零，并提供 `valid_mask`：真动作标 `True`，补齐位置标 `False`。真实动作本身也可能恰好为零，所以不能靠数值是否为零判断有效性。每个片段只能使用同一 Episode、同一 train/validation/test 划分里的动作。

## 数据与运行

使用实验 020 已有的**仿真**轨迹 `outputs/020_simulation_dataset/dataset.npz`，不把它称为真实机器人数据。运行前需先有该文件；如缺失，按 [实验 020](../020_simulation_dataset/README.md) 的步骤生成。

```bash
conda activate robot_learning
./scripts/run_tests.sh tests/test_action_chunk_dataset.py tests/test_trajectory_dataset.py
PYTHONPATH=src python experiments/044_action_chunk_alignment/run.py
```

配置见 `configs/044_action_chunk_alignment.json`。程序检查所有动作片段的 Episode、split 与步号连续性，并保存汇总和首条 Episode 边界两侧的例子到被 Git 忽略的 `outputs/044_action_chunk_alignment/results.json`。

## 结果与边界

实际运行读取 40 条仿真 Episode、`12,102` 条转移，生成同样多的起点片段，没有丢弃末尾样本。其中 `11,982` 个片段有完整 4 个动作，`120` 个片段需要补齐，总补齐槽位 `240` 个。这与每条 Episode 最后 3 个起点分别缺少 `1、2、3` 个动作相符：`40 × (1+2+3) = 240`。

边界抽查：Episode 0 最后一行是源数据第 `399` 行，动作来源索引为 `[399, -1, -1, -1]`，掩码为 `[True, False, False, False]`；第 `400` 行已经是 Episode 1 的第 0 步，且属于另一 split，未被拼到前一片段。程序对所有 `12,102` 个片段逐一检查同 Episode、同 split 和步号连续。新增动作片段测试与原轨迹数据测试共 `13 passed`。

本实验不会训练策略，也不会验证相机帧和机器人动作的真实时间戳同步；行号对齐只在这个仿真数据集的采集协议内有意义。补齐位置必须在训练损失中按掩码排除，本实验还没有训练模型或检验该损失。
