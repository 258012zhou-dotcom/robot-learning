# 实验 033：表格型 Q-Learning

## 目标

验证 Q-Learning 的一步更新与探索的作用。环境只有两个非终止状态，各有两个离散动作：

```text
S0 --take_one / 奖励 1--> 结束
S0 --continue / 奖励 0--> S1
S1 --take_zero / 奖励 0--> 结束
S1 --take_two / 奖励 2--> 结束
```

`γ=0.9`，所以起点立刻拿 1 的真实动作价值为 `1`；先前进再拿 2 的真实动作价值为 `0+0.9×2=1.8`。

## 先看什么

- 训练访问次数：智能体有没有试过 `continue` 和 `take_two`？
- Q 表：学到的起点动作价值是否接近 `1` 和 `1.8`？
- 独立评价：训练后不探索，只执行贪心策略，能否得到回报 `1.8`？

对照条件使用相同的零初始化、学习率、折扣和 1000 个 Episode，分别设置 `ε=0` 与 `ε=0.2`。相同 seed 只保证各条件可复现，不代表探索率比较具有跨随机种子的统计结论。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
PYTHONPATH=src python -m pytest -q tests/test_q_learning.py
PYTHONPATH=src python experiments/033_tabular_q_learning/run.py
python -m json.tool outputs/033_tabular_q_learning/results.json
```

输出保存在 `outputs/033_tabular_q_learning/`，不提交 Git。

## 结果

固定 seed 33、训练 1000 个 Episode 后：

| 探索率 | S0 两动作访问数 | S1 两动作访问数 | 学到的 `Q(S0)` | 独立贪心回报 |
| ---: | ---: | ---: | ---: | ---: |
| `ε=0` | 1000 / 0 | 0 / 0 | 1.0 / 0.0 | 1.0 |
| `ε=0.2` | 208 / 792 | 89 / 703 | 1.0 / 1.8 | 1.8 |

动作顺序分别为 S0 的 `take_one / continue` 和 S1 的 `take_zero / take_two`。完全不探索时，零初始化和固定的并列取首个动作，使策略一直选择 `take_one`，因此不能获得另一条路线的证据。加入探索后，S1 的 `take_two` 被访问，奖励 2 逐渐通过 `max Q(S1,·)` 传回 S0，贪心策略最终选择 `continue → take_two`。

相关 Q-Learning 与 TD 测试共 11 项通过。这里没有用多 seed 测量探索率的稳定性，也没有在连续动作点机器人上训练。
