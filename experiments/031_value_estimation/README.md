# 实验 031：V、Q、Advantage 与 Monte Carlo 估计

## 目标

使用一个能够手工计算真实答案的随机 MDP，区分单次轨迹 Return、状态价值 `V`、动作价值 `Q` 和优势 `Advantage`，再观察 Monte Carlo 样本均值如何接近真实期望。

## MDP 设计

初始状态有两个动作，执行后 Episode 最多两步结束：

- `safe`：立即奖励 1，然后结束。
- `risky`：立即奖励 0；下一步有 75% 得到 4，25% 得到 -4，然后结束。
- 折扣因子：`γ=0.9`。
- 当前策略：75% 选择 `safe`，25% 选择 `risky`。

真实动作价值可以直接计算：

```text
Q(s,safe)  = 1
Q(s,risky) = 0 + 0.9 × (0.75 × 4 + 0.25 × -4) = 1.8
```

然后根据当前策略计算：

```text
V(s) = Σ π(a|s)Q(s,a)
A(s,a) = Q(s,a) - V(s)
```

## 为什么还要采样

真实机器人通常不知道精确转移概率，也不能直接写出真实 Q。实验按当前策略采样 Episode，把同一动作产生的 Return 求平均，得到 Monte Carlo Q 估计；把所有 Episode 的 Return 求平均，得到 V 估计。

一次 Return 只是一个样本。采样量增加通常会减小估计误差，但有限样本不保证误差每次单调下降。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
PYTHONPATH=src python -m pytest -q tests/test_value_estimation.py
PYTHONPATH=src python experiments/031_value_estimation/run.py
python -m json.tool outputs/031_value_estimation/results.json
```

## 当前状态

实验已完成，回报和价值相关测试共 17 项通过。

手工计算的真实结果：

| 量 | safe | risky |
| --- | ---: | ---: |
| `Q(s,a)` | 1.0 | 1.8 |
| `A(s,a)` | -0.2 | 0.6 |

当前策略的真实状态价值为 `V(s)=1.2`，策略概率加权后的 Advantage 为 0，符合 Advantage 相对策略平均水平的定义。

Monte Carlo 采样结果：

| Episode 数 | safe/risky 样本数 | `V` 样本均值 | `Q_safe` | `Q_risky` | `V` 绝对误差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 8 / 2 | 0.800 | 1.000 | 0.000 | 0.400 |
| 100 | 77 / 23 | 0.878 | 1.000 | 0.470 | 0.322 |
| 1000 | 755 / 245 | 1.169 | 1.000 | 1.690 | 0.031 |
| 10000 | 7455 / 2545 | 1.212 | 1.000 | 1.835 | 0.012 |

少量样本时，`risky` 的 Q 甚至被估计为 0，使其 Advantage 符号从真实的正数错误地变成负数。10000 次采样后各项估计接近真实值。这说明策略梯度使用估计 Advantage 时会受到采样方差影响，但不保证每次增加一点样本都让误差严格下降。
