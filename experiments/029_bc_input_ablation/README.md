# 实验 029：BC 输入消融

## 目标

实验 028 发现，完整观察 BC 会根据速度改变动作，而 P 控制专家只使用目标误差。本实验只改变 BC 的输入特征，检验这种非必要的速度依赖是否来自训练轨迹中的相关性。

## 公平比较

- 完整观察 BC：`[position, velocity, target, target_error]`。
- 消融 BC：只输入 `[target_error]`。
- 两者使用相同专家数据、Episode 划分、随机种子、隐藏层、训练轮数和优化参数。
- 参数数量会因输入维度改变而略有不同，这是删除输入连接的直接结果。
- 使用相同未见 seed 比较测试 MSE、正常闭环、受控扰动和速度敏感性。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
PYTHONPATH=src python -m pytest -q tests/test_behavior_cloning.py tests/test_distribution_shift.py tests/test_gymnasium_rollout.py
PYTHONPATH=src python experiments/029_bc_input_ablation/train.py
PYTHONPATH=src python experiments/029_bc_input_ablation/evaluate.py
python -m json.tool outputs/029_bc_input_ablation/comparison_results.json
```

结果文件和 checkpoint 写入 `outputs/029_bc_input_ablation/`，不提交 Git。

## 当前状态

正式运行已完成，相关测试共 35 项通过。

| 策略 | 测试 MSE | 速度动作最大跨度 | 正常成功率 | 20 步扰动成功率 |
| --- | ---: | ---: | ---: | ---: |
| P 控制专家 | 不适用 | 0 | 100% | 79% |
| 完整观察 BC | 0.000006648 | 0.3257 | 100% | 100% |
| 仅目标误差 BC | 0.000000518 | 0 | 100% | 81% |

仅目标误差 BC 保留了正常闭环能力，测试 MSE 约为完整 BC 的 `1/12.8`，而且固定误差时不再随速度改变动作。它在 20 步扰动中的表现也从完整 BC 的 100% 回到接近专家的 81%。这说明完整 BC 在该条件下的额外恢复能力主要来自训练轨迹中的速度相关性，而不是准确复现了专家规则。

40 步扰动下三种策略均为 0% 成功，输入消融没有改变这个共同能力边界。
