# 实验 050：同一片段 BC 模型的时间集成闭环对照

## 目标与原理

复用实验 046 的非 ACT 片段 BC checkpoint，保持仿真环境、模型、归一化、评价 seed 均相同，只改变执行策略：每步重规划仅用最新片段第一动作，或每步重规划后将旧片段对**当前绝对时刻**的有效预测加权融合。时间集成的衰减预先固定为 `ln(2)`，不给这 40 条评价轨迹调参。

本次对照只隔离时间集成这一机制；模型仍是 MLP 序列 BC，**不是 ACT 模型的闭环评价，更不是论文复现**。

## 验证方式

单元测试用可辨认的假模型检查跨片段时间索引和 Episode 重置。闭环运行前核对 checkpoint 的来源数据哈希、评价 seed 与示范 seed 不重叠；运行后核对新跑的“最新第一动作”基线与实验 046 的逐 Episode 结果一致，两种策略面对相同初始目标。

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_sequence_behavior_cloning.py tests/test_temporal_ensemble.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/050_temporal_ensemble_closed_loop/run.py
```

配置见 `configs/050_temporal_ensemble_closed_loop.json`，结果写入 Git 忽略的 `outputs/050_temporal_ensemble_closed_loop/`。需要实验 020 数据、实验 046 checkpoint 及其闭环结果。

## 实际结果与边界

2026-09-27，使用实验 046 的同一 checkpoint，仿真 seed 1000–1039，无示范 seed 重叠；“最新第一动作”基线逐 Episode 复现了实验 046。两种策略初始目标一致，并在每条 Episode 前清空策略状态。

| 指标 | 每步只取最新第一动作 | 时间集成，`decay=ln(2)` |
| --- | ---: | ---: |
| 成功 | 40/40 | 40/40 |
| 平均终点距离 | 0.04068 m | 0.03936 m |
| 平均步数 | 223.40 | 223.25 |
| 平均累计回报 | −72.556 | −72.453 |
| 平均相邻动作指令绝对变化 | 0.005128 | 0.005120 |

逐 Episode 比较：时间集成的终点距离在 28/40 条较小，平均只低 `0.00133 m`；动作变化度量在 25/40 条较小，平均仅低 `0.00000775`。两组都达到成功率天花板，差距很小，**不能据此声称时间集成带来稳定闭环收益**。相邻动作指令的绝对变化也不等于物理机器人运动平稳性，更不是安全证明。

这是同一训练种子、同分布一维仿真任务；没有独立训练种子、扰动任务或 ACT 模型。本实验只证明缓存、索引与执行闭环可以工作，并给出当前协议下的有限对照证据。
