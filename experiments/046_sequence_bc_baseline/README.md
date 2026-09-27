# 实验 046：最小动作片段 BC 与单步 BC 对照

## 目标与原理

给同一个仿真观察预测未来 `H=4` 个电机动作，并比较两种执行方式：整段执行完再重新预测（`hold_chunk`），或每步用新观察重新预测并只执行新片段第一步（`replan_each_step`）。两者用同一个模型和同一组评价 seed；参照物是实验 027 已训练的单步 BC。这个 MLP **没有 Transformer、CVAE 或时间集成，不是 ACT**。

训练样本来自实验 020 的仿真 P 控制专家轨迹。实验 044 的 Episode 内片段和有效掩码用于监督训练；补齐的动作不计入 MSE。观察归一化只由训练 Episode 拟合，验证 Episode 选最佳轮次，测试 Episode 只在训练完成后评估。

## 验证方式

新增单元测试检查掩码、输出形状/限幅和两种执行方式的重置。实际运行再检查独立进程能读取 checkpoint，评价 seed 不与示范数据重叠，三种策略面对相同目标，并记录逐 Episode 成功与失败。

```bash
/home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_sequence_behavior_cloning.py tests/test_action_chunk_dataset.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/046_sequence_bc_baseline/run.py
```

配置见 `configs/046_sequence_bc_baseline.json`；输出位于不提交 Git 的 `outputs/046_sequence_bc_baseline/`。运行前需有实验 020 数据与实验 027 单步 BC checkpoint。

## 实际结果与适用范围

2026-09-27，CPU、训练 seed 42、动作片段长度 4、100 个 Epoch。仅使用实验 020 的 P 控制专家数据：train 14 条 Episode / 3276 个起点，validation 3 条 / 709 个起点，test 3 条 / 723 个起点。最佳验证轮次为第 89 轮；test 有效动作槽位 MSE 为 `8.04e-6`，只看片段第一步的 test MSE 为 `6.70e-6`。后者与实验 027 的单步 BC test MSE `6.65e-6` 较接近，但模型与训练目标不同，不能只凭这两个数字定优劣。

同一组未见示范的仿真环境 seed 1000–1039、相同目标下闭环：

| 策略 | 成功 | 平均终点距离 | 平均步数 | 平均累计回报 |
| --- | ---: | ---: | ---: | ---: |
| 单步 BC（实验 027 checkpoint） | 40/40 | 0.0396 m | 223.2 | −72.40 |
| 片段 BC，整段执行 | 40/40 | 0.0399 m | 223.3 | −72.49 |
| 片段 BC，每步重规划 | 40/40 | 0.0407 m | 223.4 | −72.56 |

这是**仿真同分布任务**，不是 ALOHA 或 PIPER 真机。原有单步 BC 已是 40/40，存在成功率天花板；本轮没有证明分块更好，也没有制造出足以比较纠偏能力的扰动。片段模型参数 4740 个，单步 BC 4545 个；两者训练目标不同，因此不是严格的“只改分块长度”消融。当前评价没有时间集成；`replan_each_step` 只采用每次新片段的第一步。成果是建立了可运行、可核对的序列策略基线，ACT 的 Transformer/CVAE 仍待学习和实现。
