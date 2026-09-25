# 实验 034：策略梯度与价值基线

## 目标

用一个状态、两个动作的任务观察策略梯度更新：低奖励动作得 1 分，高奖励动作得 2 分。策略初始对两个动作各给 50% 概率，精确期望回报为 1.5。

训练时从当前策略采样动作，用 `loss = -log π(a) × (reward - baseline)` 更新动作概率。这里的 `baseline` 是可手算的当前状态价值 `V = 1 + P(high)`；它只用于构造优势，不在本实验训练 Critic。

## 要核对的量

- 训练前后 `P(high)`：是否逐渐提高？
- 精确期望回报 `1 + P(high)`：是否随之提高？
- 初始 50/50 策略下，使用和不使用基线时，单样本梯度的均值与方差有何区别？

这个两动作任务用来检查策略梯度的方向和基线作用。实际机器人状态、连续动作、价值网络、PPO 裁剪都不在本实验范围内。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
PYTHONPATH=src python -m pytest -q tests/test_policy_gradient.py
PYTHONPATH=src python experiments/034_policy_gradient/run.py
python -m json.tool outputs/034_policy_gradient/results.json
```

生成结果存入 `outputs/034_policy_gradient/`，不提交 Git。

## 结果

seed 34、SGD 学习率 0.1、采样更新 200 次后：

| 完成更新 | `P(high)` | 精确期望回报 |
| ---: | ---: | ---: |
| 0 | 0.5000 | 1.5000 |
| 10 | 0.5647 | 1.5647 |
| 50 | 0.7379 | 1.7379 |
| 100 | 0.8589 | 1.8589 |
| 200 | 0.9455 | 1.9455 |

在初始策略 `P(high)=0.5` 处，枚举两个动作可精确计算单样本梯度：

| 梯度估计 | 均值 | 单样本方差 |
| --- | ---: | ---: |
| 不减基线 | 0.25 | 0.5625 |
| 减去真实 `V=1.5` | 0.25 | 0 |

两种方式在这个状态下有相同的期望更新方向；基线降低了采样波动。方差恰好为 0 只发生在这个对称、确定性的初始设置，不能推广为真实任务中加入基线即可消除方差。相关策略梯度与 Q-Learning 测试共 8 项通过。
