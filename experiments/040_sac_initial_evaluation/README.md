# 实验 040：SAC 首次独立闭环评价

## 目的与边界

验证“能运行 SAC 更新”是否已经转化为点机器人仿真中的闭环能力。这是一个 **240 步训练的早期教学检查点**，不是 SAC 论文复现，也不是算法性能结论，更不涉及真机。

## 固定协议

- 环境：`PointRobotReachEnv`，4 维观测、连续动作 `[-1, 1]`；每回合最多 200 步，目标距离范围 0.5–2.0 m。奖励、成功条件和其他设置见 [`configs/040_sac_initial_evaluation.json`](../../configs/040_sac_initial_evaluation.json)。本环境设置不同于实验 039，不能直接横向比较成功数。
- 训练：PyTorch 种子 26；仿真交互恰好 240 步，前 64 步随机预热，之后每步用当前 Actor 采样动作并做一次更新；回放池容量 1000，batch 64，固定 `alpha=0.2`。训练从环境 seed 26 开始，回合结束后依次加 1。按这次实际运行，完成 1 个回合，第二个回合只采集了前 40 步。两个训练目标分别是约 `+1.237 m`、`-1.547 m`。
- 评价：seed `100026–100045`，与可能使用的训练 seed 不重叠。训练前 SAC、训练后 SAC 都用 `tanh(高斯均值)` 的确定性动作；P 控制器用 `clip(0.5 × 目标误差)`。三者在完全相同的 20 个目标任务上运行到终止或超时。每条记录保留目标、成功、回报、步数和终点状态。
- 判定：成功由环境同时检查位置误差不超过 0.05 m、速度绝对值不超过 0.10 m/s；否则到 200 步记为超时。损失值不是成功指标。

## 运行与证据

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_sac_evaluation.py tests/test_sac_online.py tests/test_sac_training.py tests/test_sac_models.py tests/test_sac_dataflow.py tests/test_point_robot_reach_env.py
PYTHONPATH=src python experiments/040_sac_initial_evaluation/run.py
```

本次实际运行：相关测试 `80 passed, 2 warnings`，警告是观测空间使用无限边界的既有 Gymnasium 提示；仿真程序正常结束，记录训练 `240` 步、`176` 次梯度更新。忽略 Git 的 [`outputs/040_sac_initial_evaluation/results.json`](../../outputs/040_sac_initial_evaluation/results.json) 保存汇总和配置，[`episodes.csv`](../../outputs/040_sac_initial_evaluation/episodes.csv) 保存 60 条逐回合记录。

## 结果

| 策略 | 成功 | 正目标成功 | 负目标成功 | 平均原环境回报 | 平均最终距离 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练前 SAC | 4/20 | 4 | 0 | -185.23 | 0.817 m |
| 训练后 SAC | 0/20 | 0 | 0 | -331.14 | 2.607 m |
| P 控制器 | 6/20 | 1 | 5 | -67.70 | 0.140 m |

训练后的 SAC 20 条任务全部运行到 200 步超时。11 个负方向目标任务的最终位置都在原点正侧，终点约 `+2.287–+4.438 m`，与目标方向相反。正方向的 9 个任务中有 7 个在终点尚未到达目标位置；这只描述终点，不等于整条轨迹从未接近目标。训练前成功的 4 条任务，训练后均失败。P 控制器也只成功 6/20，说明此处 200 步限制与目标距离范围下，不能借用实验 039 中另一环境协议的 P 控制成绩。

## 结论与限制

本次证据支持“当前这份初始化、240 步预算和固定超参数的 SAC 更新没有改善闭环任务，反而在这 20 条独立任务上退化”。训练只完成 1 个完整回合、涉及很少目标，不足以证明 SAC 算法普遍无效，也不能把负方向失败唯一归因于奖励、熵系数、探索、数据覆盖或代码错误。下一步应先复核训练轨迹和数据覆盖，再决定是否增加预算或改协议；任何改动都应重新用独立任务评价。
