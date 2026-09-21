# 实验反思

## 当前状态

实验 027 第一轮已完成。已经完成专家数据筛选、按 Episode 隔离的数据划分、train-only 归一化、最小 MLP、离线监督训练、checkpoint 独立加载和未见 seed 的闭环对照。

当前结论只适用于与训练环境相同的动力学和观察条件。分布偏移、扰动恢复与 DAgger 属于下一模块，不能把它们算作本实验已经验证的能力。

## 2026-09-21 离线训练记录

使用 seed 42、CPU、PyTorch `2.12.1+cu130` 训练 100 个 Epoch。模型共有 4545 个可训练参数，训练、验证和测试分别使用 3276、709 和 723 条专家 Transition。

- 初始 validation MSE：`0.0488886`。
- 最佳轮次：第 91 个 Epoch。
- 最佳 validation MSE：`4.69234e-6`。
- test MSE：`6.64819e-6`。
- 独立 Python 进程加载 `best_model.pt` 后重新计算的 test MSE 与训练记录逐值相同，均为 `6.648194592301456e-06`。

实际运行命令：

```bash
PYTHONPATH=src python -m pytest -q tests/test_behavior_cloning.py tests/test_training.py tests/test_trajectory_dataset.py
PYTHONPATH=src python experiments/027_behavior_cloning/run.py
```

相关测试共 28 项通过。结果文件状态明确记录为 `offline_training_complete_closed_loop_pending`，避免把当前检查点误写为完整实验成功。

加入 checkpoint 格式、严格参数恢复和单步策略适配测试后，相关测试增加到 31 项并全部通过。checkpoint 包含模型结构、全部参数、训练集归一化统计、数据集内容哈希和 seed；加载时使用安全的 `weights_only=True`，并拒绝未知格式版本。

## 闭环结果

闭环评价使用 seed 1000–1039，与示范数据 seed 无重叠；目标包含 19 个负方向和 21 个正方向。加入真实环境策略接口测试后，相关测试共 43 项通过。

| 策略 | 成功 Episode | 平均最终距离 | 平均累计回报 | 平均步数 |
| --- | ---: | ---: | ---: | ---: |
| Random | 9 / 40 | 1.31579 | −479.883 | 366.98 |
| P 控制专家 | 40 / 40 | 0.03943 | −72.749 | 224.65 |
| Behavior Cloning | 40 / 40 | 0.03961 | −72.396 | 223.18 |

BC 的正、负方向成功率均为 100%，说明 test 数据只包含正目标的问题没有阻止它在本次同分布闭环评价中控制负目标。BC 平均比 P 控制少约 1.48 步，但逐 Episode 中 BC 更慢 15 次、更快 11 次、相同 14 次，因此不能据此声称 BC 稳定优于专家。

P 控制专家和 BC 的平均 Overshoot 都约为 `0.46 m`。BC 学到了专家的到达能力，也继承了专家明显过冲再返回的行为；这是模仿学习会复制示范策略特点的直接例子。

## 第一轮结论

当前 BC 不只是离线 MSE 较小：它在同分布、未见 seed 的闭环仿真中达到 40/40 成功，接近 P 控制专家并明显优于随机策略。证据仍不足以说明它对状态分布偏移具有恢复能力，因为质量、阻尼、观测和动作执行条件都没有变化。

## 当前认识

- BC 的离线训练与普通监督学习相同，但部署时模型动作会改变下一时刻输入。
- validation 用于选择最佳模型，test 不能参与训练轮次或超参数选择。
- P 控制数据是专家示范；混入数量更多的随机策略 Transition 会改变学习目标。
- 输入归一化属于模型的一部分，部署时必须保存并复用同一组 train 统计量。
