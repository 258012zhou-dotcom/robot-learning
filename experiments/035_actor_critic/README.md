# 实验 035：Actor-Critic 学习价值基线

## 目标

实验 034 使用了已知的真实 `V(s)`。这次让 Critic 从采样奖励学习状态价值，并用它给 Actor 计算 Advantage。任务仍是一个状态、两个动作，奖励分别为 1 和 2；相同训练预算下保留真实价值基线对照。

## 两个损失各做什么

```text
Actor loss  = -log π(a|s) × (reward - Critic(s))
Critic loss = (Critic(s) - reward)²
```

Actor 的 Advantage 使用执行动作前的 Critic 预测，更新 Actor 时不改变 Critic。Critic 用实际奖励更新自己的预测，不改变 Actor。本任务只有一步且立即终止，所以实际奖励就是完整 Return；这里没有下一状态的 Bootstrapping。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
PYTHONPATH=src python -m pytest -q tests/test_actor_critic.py tests/test_policy_gradient.py
PYTHONPATH=src python experiments/035_actor_critic/run.py
python -m json.tool outputs/035_actor_critic/results.json
```

## 观察指标

- `P(high)` 和精确期望回报：Actor 是否逐渐偏向 2 分动作？
- Critic 预测与精确 `V(s)=1+P(high)` 的差距：Critic 是否跟上不断变化的策略？
- 真实价值基线对照：在同一任务中检查学习型基线的表现边界。

输出位于 `outputs/035_actor_critic/`，不提交 Git。这个任务没有状态变化，因此 Critic 是一个可学习的标量；多状态任务才需要状态到价值的函数近似。

## 结果

固定 seed 34，两个条件都训练 200 次，每次只采样一个动作：

| 更新次数 | 真实基线 `P(high)` | 学习型 Critic `P(high)` | Critic 预测 `V` | 当前真实 `V` | 绝对误差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5000 | 0.5000 | 0.0000 | 1.5000 | 1.5000 |
| 1 | 0.5062 | 0.4875 | 0.2000 | 1.4875 | 1.2875 |
| 10 | 0.5647 | 0.5537 | 1.1629 | 1.5537 | 0.3908 |
| 50 | 0.7379 | 0.7301 | 1.6630 | 1.7301 | 0.0671 |
| 100 | 0.8589 | 0.8581 | 1.8304 | 1.8581 | 0.0276 |
| 200 | 0.9455 | 0.9437 | 1.9781 | 1.9437 | 0.0344 |

学习型 Critic 初始预测过低，第一次抽到低奖励动作时仍给出正 Advantage，因此 Actor 把高奖励动作概率暂时降到 0.4875。之后 Critic 从奖励中学习并追随不断变化的策略；到第 200 次更新时，两种条件的 `P(high)` 分别为 0.9455 和 0.9437。

这些是单 seed 的教学结果，不能用最后两组数值的接近程度证明学习型 Critic 与真实基线性能相同。Critic 的绝对误差也没有严格单调下降。相关 Actor-Critic 与策略梯度测试共 7 项通过。
