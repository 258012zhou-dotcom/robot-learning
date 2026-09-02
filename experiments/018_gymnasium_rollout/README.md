# 实验 018：Gymnasium 到达任务与控制基线

## 目标

把实验 017 的 MuJoCo 点机器人封装成 Gymnasium 环境，建立完整的
`Observation → Policy → Action → Reward → next Observation` 循环，并比较随机策略和 P 控制策略。

## 环境契约

- Observation：`[位置, 速度, 目标位置, 目标误差]`。
- Action：一个范围为 `[-1, 1]` 的电机控制量。
- Reward：`-目标距离 - 0.01 × action²`。
- Terminated：位置误差和速度同时进入成功容差。
- Truncated：尚未成功，但已经达到最大 Episode 步数。
- 一个 Action 保持 5 个 MuJoCo 物理步，因此控制周期为 `0.05 s`。

## 公平对照

两种策略使用相同的 40 个环境 seed，所以每一对 Episode 都具有相同的目标位置：

- 随机策略：不使用 Observation，在动作范围内均匀采样。
- P 控制策略：`action = clip(Kp × target_error, -1, 1)`，其中 `Kp=0.5`。

P 控制器不是学习算法，而是后续强化学习实验需要超过的传统控制基线。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh \
  tests/test_point_robot_reach_env.py \
  tests/test_gymnasium_rollout.py
PYTHONPATH=src python experiments/018_gymnasium_rollout/run.py
python -m json.tool outputs/018_gymnasium_rollout/results.json
```

## 输出

- `results.json`：策略的聚合指标。
- `episodes.csv`：每一个 Episode 的原始评估记录。
- `policy_comparison.png`：成功率、最终距离和累计奖励对比。
- `example_rollouts.png`：同一目标下两种策略的位置与动作轨迹。
- `run.log`：运行摘要。

## 指标解释

- 成功率是主要任务指标。
- 最终距离衡量 Episode 结束时距离目标还有多远。
- 累计奖励是每步奖励之和，既受运动质量影响，也受 Episode 长度影响，不能脱离其他指标单独解释。
- 平均步数只有结合成功率才有意义；失败策略跑满时限并不代表它更稳定。

## 实际结果

40 个相同目标任务上的结果：

- 随机策略成功 `3/40`，成功率 `7.5%`，平均最终距离 `1.3832 m`。
- P 控制策略成功 `40/40`，成功率 `100%`，平均最终距离 `0.0365 m`。
- 随机策略平均回报为 `-541.31`，P 控制策略为 `-88.32`。
- 随机策略平均执行 `390.4` 步，接近 400 步上限；P 控制策略平均执行 `240.8` 步后成功终止。

这些结果说明简单 P 控制器是有效基线，同时也说明随机策略的成功率不一定严格为零：随机游走可能偶然同时满足位置和速度容差。
