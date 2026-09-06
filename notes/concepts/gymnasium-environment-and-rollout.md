# Gymnasium 环境、Episode 与策略评估

## 快速复习

- Gymnasium 环境把任务统一成 `reset()` 和 `step(action)` 两个主要接口。
- Observation 是策略能看到的信息；Action 是策略发给环境的控制量。
- Reward 是每一步的优化信号，不一定等同于最终评价指标。
- Episode 是从重置到终止或截断的一条完整交互轨迹。
- `terminated` 表示任务或自然终止条件成立；`truncated` 表示时间上限等外部限制结束了轨迹。
- 一个可靠实验需要明确空间、数据类型、控制周期、结束条件、随机种子和评估指标。
- 随机策略用于检查最低基线，传统控制器用于建立有意义的任务基线。

## 环境交互循环

Gymnasium 把不同机器人任务抽象为同一种循环：

```text
observation, info = environment.reset(seed=seed)

while not (terminated or truncated):
    action = policy(observation)
    observation, reward, terminated, truncated, info = environment.step(action)
```

各组件的职责应当分开：

- Environment：维护状态、推进物理、计算奖励和判断结束。
- Policy：只根据当前 Observation 生成 Action。
- Rollout runner：连接策略与环境，并记录完整轨迹。
- Evaluator：汇总多条轨迹，但保留逐 Episode 的原始证据。

这种分层让同一个环境可以测试随机策略、传统控制器和学习策略，也让同一个策略接口可以迁移到其他环境。

## Observation 与 observation space

Observation 是策略实际收到的输入，不必等于仿真器的全部内部状态。实验 018 使用：

```text
[position, velocity, target_position, target_error]
```

其中 `target_error = target_position - position`。`observation_space` 声明形状、范围和数据类型，用于：

- 检查环境输出是否符合契约。
- 告诉算法输入维度和合理范围。
- 防止训练过程中悄悄出现错误 shape 或 dtype。

当前位置、速度和目标误差的空间上下界设为无穷，目标位置仍有限。MuJoCo 的关节限位是软约束，瞬间状态可能略超出 MJCF 范围；观测还可能加入位置偏差，不能把关节范围当成测量值的硬边界。Gymnasium checker 的无穷边界提醒不等于契约失败。训练时可以明确增加归一化或裁剪 Wrapper，但不能悄悄裁剪真实状态来掩盖空间声明错误。

输入边界同样属于环境契约：非有限动作、初始位置和目标必须报错。当前实现先验证 `reset` 选项，再改变物理状态、延迟队列和随机数状态；非法 `step` 不推进时间、不改变队列。测试除了检查异常，还与未收到非法输入的参考环境继续运行并比较结果。

## Action 与 action space

实验中的 Action 是一个连续电机输入：

```text
action.shape == (1,)
action ∈ [-1, 1]
```

Action 表示执行器控制量，不是目标位置。环境会把越界输入裁剪到 MJCF 执行器范围，但策略本身也应产生合法动作；长期依赖环境裁剪会隐藏策略输出异常。

连续控制中还要区分三个时间尺度：

- MuJoCo timestep：一次物理积分的时间。
- frame skip：一个 Action 保持多少个物理步。
- control timestep：策略两次决策之间的仿真时间。

实验 018 中：

```text
physics timestep = 0.01 s
frame skip       = 5
control timestep = 0.05 s
control rate     = 20 Hz
```

改变 frame skip 会同时改变控制频率和每个动作影响系统的时长，不能把它只看成性能参数。

## Reward 与任务指标

实验奖励为：

```text
每步奖励 = −abs(目标位置 − 当前真实位置) − 0.01 × 动作²
```

第一项鼓励靠近目标，第二项轻微惩罚过大的动作。由于每一步奖励大多为负，累计奖励越接近零通常越好。

Reward 是训练时逐步优化的信号，而成功率、最终距离等是实验评价指标。两者相关但不完全相同：

- Episode 越长，累计的负奖励通常越多。
- 奖励设计不当时，策略可能提高 Reward 却没有真正完成任务。
- 因此不能只报告 Reward，必须同时报告任务指标和失败案例。

## terminated 与 truncated

实验成功要求同时满足：

```text
distance <= 0.05 m
abs(velocity) <= 0.10 m/s
```

位置接近但高速穿过目标不能算稳定到达。

- `terminated=True`：成功条件成立，任务本身结束。
- `truncated=True`：400 步耗尽但任务未成功，是时间上限截断。

这是当前环境的具体约定，不是说两个标记必须互斥。外层 `TimeLimit` 可能在成功的同一步达到时间上限，此时二者均为 `True`。采集器应按 `terminated or truncated` 停止，并分别原样保存；学习算法仍需根据 `terminated` 判断任务是否真正终止。

强化学习计算回报或 bootstrap target 时，两者的语义不同，不能简单合并成一个没有含义的 `done`。

## Episode 数据形状

如果一个 Episode 执行了 `T` 个动作：

```text
observations.shape == (T + 1, observation_dimension)
actions.shape      == (T, action_dimension)
rewards.shape      == (T,)
```

多出来的一个 Observation 是执行第一个 Action 之前的初始状态。记录轨迹时最常见的错误之一，就是把 Observation 与 Action 强行保存成相同长度并丢失初始状态。

## 随机性与公平比较

`reset(seed=...)` 控制环境随机数，例如目标位置。实验 018 让随机策略和 P 控制策略逐项使用相同的 40 个环境 seed，因此两者面对相同目标。

随机策略还拥有自己的随机数生成器。环境 seed 和策略 seed 应分开管理，因为它们控制不同来源的随机性。

公平比较至少需要保证：

- 相同任务分布和测试 seed。
- 相同最大 Episode 步数和成功条件。
- 相同 Observation、Action 和动力学。
- 报告多次 Episode 的统计量，而不是挑选一次运行。

## 实验 018 的基线结果

随机策略不读取 Observation；P 控制策略使用：

```text
误差 e = 目标位置 − 观测位置
动作 = clip(0.5 × e, −1, 1)
```

`clip(value, low, high)` 把数值限制在给定区间。例如误差为 3 时，原始动作是 1.5，限幅后执行 1.0。

40 个相同目标任务的结果：

| 策略 | 成功率 | 平均最终距离 | 平均累计奖励 | 平均步数 |
| --- | ---: | ---: | ---: | ---: |
| Random | 7.5% | 1.3832 m | -541.31 | 390.4 |
| P control | 100% | 0.0365 m | -88.32 | 240.8 |

P 控制轨迹先越过目标，再振荡衰减并稳定。它只使用位置误差，没有显式速度反馈；到达目标附近时机器人仍有惯性。随机策略的 3 次成功属于随机轨迹偶然进入成功区域，不代表它学会了任务。

## 当前边界与下一步

- 实验 018 的基准任务是一维、无噪声、完整状态可见的仿真任务；这不是后续所有实验的统一假设。
- P 控制器直接获得目标误差，没有视觉感知或状态估计误差。
- 后续实验 021 已加入动力学参数随机化，022 已模拟执行延迟与观测偏差。复杂接触、障碍物与真实硬件延迟尚未通过这些实验验证。
- 尚未训练策略；后续学习算法应至少与随机策略和 P 控制基线比较。
- 控制频率、轨迹采集与领域随机化已有对应笔记；接下来复用这些接口学习策略训练，不需要重新搭建环境。
