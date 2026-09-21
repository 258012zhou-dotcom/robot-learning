# 实验 027：Behavior Cloning

## 目标

使用实验 020 中由 P 控制器生成的专家轨迹，训练一个最小 MLP 策略学习 `observation → action`，并比较离线动作误差与后续闭环任务表现。

当前已完成专家数据筛选、train-only 归一化、离线监督训练、checkpoint 独立进程加载验证，以及未见 seed 上的随机策略、P 控制器与 BC 闭环对照。

## 数据与模型

- 输入：`[position, velocity, target_position, target_error]`。
- 标签：P 控制器的单维电机动作；随机策略数据不参与训练。
- 划分：14 个 train、3 个 validation、3 个 test 专家 Episode。
- 模型：`4 → 64 → 64 → 1` MLP，隐藏层使用 ReLU，输出使用 `tanh` 限制到环境动作范围。
- 损失：动作均方误差（MSE）。

归一化均值和标准差只由 train 专家观察计算。validation 用于选择最佳训练轮次，test 只在训练结束后计算一次离线 MSE。

## 运行当前检查点

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
PYTHONPATH=src python -m pytest -q tests/test_behavior_cloning.py tests/test_training.py tests/test_trajectory_dataset.py
PYTHONPATH=src python experiments/027_behavior_cloning/run.py
python -m json.tool outputs/027_behavior_cloning/offline_results.json
PYTHONPATH=src python experiments/027_behavior_cloning/evaluate.py
python -m json.tool outputs/027_behavior_cloning/closed_loop_results.json
```

当前输出：

- `best_model.pt`：最佳验证轮次的模型参数、归一化参数和模型结构配置。
- `offline_results.json`：数据数量、训练曲线及离线 MSE。
- `closed_loop_results.json`：三种策略的逐 Episode 结果和聚合指标。
- `run.log`：运行摘要。

这些输出位于 `outputs/`，不提交 Git。离线 MSE 较小不等于闭环成功，必须等后续策略 Rollout 后才能得出实验结论。

## 当前离线结果

2026-09-21 使用 seed 42 在 CPU 上完成 100 个 Epoch：

- 随机初始化时 validation MSE：`0.0488886`。
- 最佳轮次：第 91 个 Epoch。
- 最佳 validation MSE：`4.69234e-6`。
- 最终 test MSE：`6.64819e-6`。
- 新进程加载 checkpoint 后重新计算的 test MSE：`6.64819e-6`，与训练进程记录值完全一致。

这些数值只证明模型能在保留的专家状态上预测接近专家的动作，不证明闭环到达成功率。

## 闭环评价结果

使用与示范数据无重叠的 seed 1000–1039，共 40 个 Episode；其中 19 个负方向目标、21 个正方向目标。三种策略面对完全相同的目标。

| 策略 | 成功率 | 平均最终距离 | 平均累计回报 | 平均步数 |
| --- | ---: | ---: | ---: | ---: |
| Random | 22.5% | 1.31579 | −479.883 | 366.98 |
| P 控制专家 | 100% | 0.03943 | −72.749 | 224.65 |
| Behavior Cloning | 100% | 0.03961 | −72.396 | 223.18 |

BC 在这组同分布、未见 seed 的仿真任务中复现了专家的任务成功率，正负方向均为 100%。这证明当前最小数据、训练、保存加载和闭环执行链路成立；不证明它能处理动力学变化、观测偏差、动作延迟或训练分布之外的状态。

P 控制器与 BC 的平均 Overshoot 分别约为 `0.45995 m` 和 `0.45707 m`，两者都存在明显越过目标后再返回的行为。随机策略的平均 Overshoot 更小不能解释为更好，因为多数随机 Episode 根本没有到达目标。
