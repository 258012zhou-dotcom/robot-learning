# 实验 036：PPO 裁剪目标的最小对照

## 先看问题

实验 035 让 Actor 使用 Critic 估计的 Advantage 更新，但没有限制重复使用同一批数据时策略可以改变多少。本实验固定旧策略和数据，对比普通重要性加权目标与 PPO 裁剪目标，观察二者如何继续推动好动作的概率。

## 机制

同一个状态有两个动作：低奖励为 1，高奖励为 2。旧策略对两个动作各给 0.5 概率，精确旧价值为 1.5，所以两动作的 Advantage 分别为 `-0.5` 和 `+0.5`。这里直接枚举两个动作各一次，排除采样噪声；这不是实际环境 Rollout。

```text
ratio = 新策略对已记录动作的概率 / 旧策略对该动作的概率
普通目标 = mean(ratio × Advantage)
裁剪目标 = mean(min(ratio × Advantage,
                    clip(ratio, 1-ε, 1+ε) × Advantage))
```

两组从相同策略出发，以相同学习率重复使用固定数据。`ε=0.2` 时，正 Advantage 的高奖励动作在 `P(high)>0.6` 后，不再从这批数据获得继续增大概率的梯度。裁剪的是目标，不是给概率施加硬上限；一次更新仍可能越过 0.6。

## 运行与验证

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
PYTHONPATH=src python -m pytest -q tests/test_ppo_clipping.py
PYTHONPATH=src python experiments/036_ppo_clipping/run.py
python -m json.tool outputs/036_ppo_clipping/results.json
```

观察 `P(high)` 和高奖励动作的 `ratio_high`。程序结果写入忽略 Git 的 `outputs/036_ppo_clipping/`。本实验只验证裁剪目标的局部机制，不包含采样 Rollout、学习型 Critic、GAE、批量重采集或闭环成功率；不能称作 PPO 算法复现。

## 结果

同一批两动作数据、学习率 1.0、`ε=0.2`，结果如下。这里没有随机采样，因此不涉及 seed 比较。

| 更新次数 | 普通目标 `P(high)` | 裁剪目标 `P(high)` | 裁剪组的 `ratio_high` |
| ---: | ---: | ---: | ---: |
| 0 | 0.5000 | 0.5000 | 1.0000 |
| 1 | 0.5622 | 0.5622 | 1.1244 |
| 2 | 0.6216 | 0.6216 | 1.2431 |
| 5 | 0.7598 | 0.6216 | 1.2431 |
| 10 | 0.8707 | 0.6216 | 1.2431 |
| 40 | 0.9715 | 0.6216 | 1.2431 |

第二次更新一步跨过了 `ratio_high=1.2`；裁剪不会把概率拉回范围内，但之后这批数据不再提供继续增大概率的梯度。普通目标继续把概率推到 0.9715。这只能说明这个固定批次上的裁剪机制，不能证明 PPO 在环境中训练效果更好。相关策略梯度、Actor-Critic 与裁剪目标测试共 12 项通过。
