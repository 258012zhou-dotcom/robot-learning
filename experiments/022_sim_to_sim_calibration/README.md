# 实验 022：Sim-to-Sim 失配校准与迁移评价

## 目标

把一个带隐藏执行器和传感器失配的 MuJoCo 环境当作“目标系统”，验证直接迁移为何退化，以及少量校准数据能否改善新任务表现。

这是 Sim-to-Sim Transfer Proxy，不是真实硬件 Sim-to-Real 结果。它先建立校准数据、参数估计和保留集评价的正确流程。

## 隐藏失配

目标环境包含：

- 执行器增益小于 1：实际动作弱于策略命令。
- 若干控制步动作延迟：当前命令稍后才生效。
- 固定位置观测偏置：策略看到的位置与真实位置不同。

隐藏真值由实验运行框架保存，但估计函数只能读取命令和策略可见的 Observation，不能读取环境 `info` 中的真值。

## 校准方法

机器人从已知零位出发，持续发送一个合法的小动作：

1. 初始测量位置减去已知零位，估计位置偏置。
2. 找到速度第一次超过阈值的控制步，估计动作延迟。
3. 将目标系统第一次响应速度除以 nominal 仿真的第一次响应速度，估计动作增益。

第三步依赖一个重要假设：质量和阻尼已经匹配。如果多种动力学误差同时存在，一次阶跃响应不能唯一确定所有参数。

## 对比方法

- `direct_nominal`：原 PD 控制器直接部署，不处理失配。
- `estimated_calibration`：使用校准估计修正位置偏置、执行器增益，并根据延迟做短期位置预测。
- `oracle_calibration`：使用隐藏真值补偿，只表示理想估计上界。

三种方法在校准完成后冻结，并使用相同的 60 个新任务 seed 评价。成功判定和最终误差使用真实位置，而不是带偏置的测量位置。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh \
  tests/test_point_robot_reach_env.py \
  tests/test_sim_to_sim_calibration.py
PYTHONPATH=src python experiments/022_sim_to_sim_calibration/run.py
python -m json.tool outputs/022_sim_to_sim_calibration/results.json
```

## 输出

- `calibration_trace.csv`：校准命令与测量响应。
- `calibration_response.png`：速度首次响应与延迟估计。
- `evaluation_episodes.csv`：每条保留集 Episode 的真实评价。
- `transfer_evaluation.png`：成功率、耗时与最终真实误差对比。
- `transfer_example.png`：同一目标上的真实轨迹和策略命令。
- `results.json`：参数估计误差与汇总指标。
- `run.log`：运行摘要。

## 本次结果

隐藏目标域参数为动作增益 `0.65`、动作延迟 `3` 步、位置偏置 `0.08 m`。一次校准轨迹得到：

- 动作增益：`0.6500000564`。
- 动作延迟：`3` 步。
- 位置偏置：`0.0799999982 m`。

在 60 个未参与校准的新目标上：

| 方法 | 成功率 | 平均耗时 | 平均最终真实误差 | 平均真实超调 |
| --- | ---: | ---: | ---: | ---: |
| direct nominal | 41.67% | 13.275 s | 0.0670 m | 0.0629 m |
| estimated calibration | 100% | 3.923 s | 0.0485 m | 0 m |
| oracle calibration | 100% | 3.923 s | 0.0485 m | 0 m |

估计方案与 oracle 几乎重合，是因为本实验使用固定偏置、无噪声测量，而且除了增益和延迟以外的动力学已经匹配。这是对校准流程的受控验证，不代表真实硬件上也能通过一条轨迹精确辨识。

## 当前边界

- 目标系统仍是 MuJoCo，不包含真实硬件接触、温度、通信抖动和安全风险。
- 只使用一次无噪声阶跃校准，现实中应重复测量并报告不确定性。
- 延迟补偿只采用匀速短期预测，不是完整的模型预测控制。
- 不能从本实验声称已经完成真实 Sim-to-Real。
