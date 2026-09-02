# 实验 019：控制频率对闭环行为的影响

## 目标

保持 MuJoCo 模型、目标任务、P 控制增益和最长仿真时间不变，只改变 `frame_skip`，观察控制频率如何影响完成时间、超调和策略决策次数。

## 实验变量

MuJoCo 物理步长固定为 `0.01 s`：

| frame skip | 控制周期 | 控制频率 | 20 秒最多决策次数 |
| ---: | ---: | ---: | ---: |
| 1 | 0.01 s | 100 Hz | 2000 |
| 5 | 0.05 s | 20 Hz | 400 |
| 10 | 0.10 s | 10 Hz | 200 |
| 20 | 0.20 s | 5 Hz | 100 |

每一种频率都使用相同的 40 个环境 seed 和 `Kp=0.5`，因此目标位置逐项相同。最大 Episode 步数随控制频率改变，以保证每组最长仿真时间都是 20 秒。

## 指标

- 成功率：是否在位置和速度容差内稳定到达。
- 完成时间：Episode 执行步数乘以控制周期。
- 位置超调：沿目标方向越过目标的最远距离。
- 决策次数：策略实际计算 Action 的次数。
- 时间积分奖励：`Σ reward × control_timestep`，用于消除每秒奖励次数不同造成的主要尺度差异。

原始累计奖励不能直接跨频率比较。100 Hz 每秒产生 100 个 Reward，5 Hz 每秒只产生 5 个；即使连续运动轨迹接近，前者也会累积更多负值。固定控制频率训练时可以使用逐步 Reward，跨频率分析则应考虑时间尺度。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh \
  tests/test_point_robot_reach_env.py \
  tests/test_gymnasium_rollout.py
PYTHONPATH=src python experiments/019_control_frequency/run.py
python -m json.tool outputs/019_control_frequency/results.json
```

## 输出

- `results.json`：每个控制频率的聚合结果。
- `episodes.csv`：逐 Episode 原始指标。
- `frequency_comparison.png`：成功率、完成时间、超调与决策次数。
- `example_trajectories.png`：同一目标下的位置和动作轨迹。
- `run.log`：运行摘要。

## 解释边界

本实验只说明当前一维模型和当前 P 增益下的频率影响。控制频率更高不必然在所有系统中更好，因为真实系统还受传感器频率、通信延迟、计算时间和执行器带宽限制。

## 实际结果

四种频率在 40 个相同目标上都达到 `100%` 成功率，但频率降低时出现稳定趋势：

| 频率 | 平均完成时间 | 平均超调 | 平均决策次数 | 时间积分奖励 |
| ---: | ---: | ---: | ---: | ---: |
| 100 Hz | 11.453 s | 0.5380 m | 1145.3 | -4.3276 |
| 20 Hz | 12.039 s | 0.5517 m | 240.8 | -4.4160 |
| 10 Hz | 12.965 s | 0.5693 m | 129.7 | -4.5532 |
| 5 Hz | 13.880 s | 0.6060 m | 69.4 | -4.7709 |

高频控制在当前任务中更快、超调更小，但需要更多策略推理次数；这体现了控制质量与计算开销之间的权衡。
