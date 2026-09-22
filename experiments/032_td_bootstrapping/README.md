# 实验 032：TD 与 Bootstrapping

## 目标

在一个真实价值可以手算的确定性两状态链上，对比 Monte Carlo 与一步 TD 如何学习状态价值，并观察 TD 的价值怎样从终止状态逐步向前传播。

## 环境

```text
S0 --奖励 0--> S1 --奖励 1--> Terminal
```

配置使用 `γ=0.9`、学习率 `α=0.2`，所以真实价值为：

```text
V(S1) = 1
V(S0) = 0 + 0.9 × 1 = 0.9
```

Monte Carlo 等 Episode 结束后，让两个状态分别学习完整 Return。TD(0) 在线更新：访问 S0 时只能使用当时的 `V(S1)`，然后到达 S1 才能从终止奖励继续学习。

## 公平边界

两种方法从全零价值开始，使用相同学习率和完全相同的确定性 Episode。这个设置刻意没有随机方差，因此主要展示 Bootstrapping 的价值传播，而不能用来证明 Monte Carlo 或 TD 在一般任务中谁更好。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
PYTHONPATH=src python -m pytest -q tests/test_temporal_difference.py
PYTHONPATH=src python experiments/032_td_bootstrapping/run.py
python -m json.tool outputs/032_td_bootstrapping/results.json
```

## 当前状态

实验已完成，强化学习基础相关测试共 24 项通过。

| 完成 Episode | TD `V(S0)` | TD `V(S1)` | MC `V(S0)` | MC `V(S1)` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.200 | 0.180 | 0.200 |
| 2 | 0.036 | 0.360 | 0.324 | 0.360 |
| 5 | 0.236 | 0.672 | 0.605 | 0.672 |
| 10 | 0.562 | 0.893 | 0.803 | 0.893 |
| 25 | 0.875 | 0.996 | 0.897 | 0.996 |
| 50 | 0.900 | 1.000 | 0.900 | 1.000 |

第一个 Episode 访问 S0 时，`V(S1)` 仍为 0，所以 TD 的 `V(S0)` 暂时不变；S1 从终止奖励学到价值后，后续 Episode 才逐渐把价值向 S0 传播。这就是本实验中的 Bootstrapping。

Monte Carlo 在这个确定性短链中更快，因为每次完整 Return 都是无噪声的真实目标。这个结果不能外推成 MC 普遍优于 TD；长 Episode、随机回报和在线学习会改变偏差—方差与更新速度的权衡。
