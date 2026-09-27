# 实验 051：低维 ACT 模型的仿真闭环

## 目标与原理

把实验 047 训练的低维 ACT 风格模型加载到实验 020 的一维点机器人仿真环境。每步只输入当前观察，推理使用 `z=0`，**绝不提供未来示范动作**。比较同一模型的两种执行方式：仅采用最新片段第一动作，或融合对当前绝对时刻的重叠预测。两组都固定使用 seed 1000–1039，并与实验 050 的目标列表核对。

这一步补齐教学规模模型的闭环执行证据；模型没有相机、双臂或原论文的完整结构，**不是 ACT 论文复现**。实验 050 的非 ACT 序列 BC 仅作相同环境的参考，不据此宣称算法优劣，因为模型结构、损失与训练过程不同。

## 验证方式

单元测试检查推理没有示范输入、每步查询一次、时间集成对齐、Episode 缓存清空，以及 checkpoint 与训练时归一化的重新加载。实际运行核对 checkpoint 的来源数据哈希、评价 seed 与示范数据不重叠、动作维度和两种策略的初始目标一致；记录逐 Episode 成功、终点距离、步数、累计回报、超调和相邻动作指令变化。

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_act_lowdim_policy.py tests/test_act_lowdim.py tests/test_temporal_ensemble.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/051_act_lowdim_closed_loop/run.py
```

配置见 `configs/051_act_lowdim_closed_loop.json`，结果保存在 Git 忽略的 `outputs/051_act_lowdim_closed_loop/`。依赖实验 020 数据、实验 047 checkpoint 和实验 050 的参考结果。

## 实际结果与边界

2026-09-27，实验 047 的训练 seed 42 checkpoint，40 个未见示范的仿真环境 seed 1000–1039。来源数据哈希、动作维度、与实验 050 的种子和初始目标均已核对；两条推理路径都没有输入示范动作，Episode 开始前清空缓存。

| 指标 | ACT `z=0`，最新第一动作 | ACT `z=0`，时间集成 |
| --- | ---: | ---: |
| 成功 | 40/40 | 40/40 |
| 平均终点距离 | 0.04002 m | 0.03966 m |
| 平均步数 | 224.30 | 227.85 |
| 平均累计回报 | −72.891 | −73.489 |
| 平均相邻动作指令绝对变化 | 0.005176 | 0.005101 |

时间集成在 32/40 条 Episode 上有更小的命令变化，均值约小 `0.0000750`；但平均完成步数**多 3.55 步**，累计回报也更低。终点距离均值仅差约 `0.000366 m`。两组都达到当前任务的成功率天花板，不能据此宣称哪种执行方式整体更好。完整逐 Episode 记录在 `outputs/051_act_lowdim_closed_loop/closed_loop_results.json`。

实验 050 的非 ACT 序列 BC 最新第一动作在同组 seed 上也是 40/40，仅作为任务难度与评价协议参考；两种模型的结构、训练目标和参数量不同，不能把它们的细小数值差异当作公平的 ACT-vs-BC 算法结论。这里是**一维、同分布仿真、单训练种子**，不是视觉双臂操作，也不是 ACT 论文复现；离线 L1 与这一轮 40/40 均不证明复杂任务能力或鲁棒性。
