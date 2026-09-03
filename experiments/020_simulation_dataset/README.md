# 实验 020：可复现的仿真轨迹数据集

## 目标

把 Gymnasium Episode 转换为可用于行为克隆、动力学学习和离线强化学习的 Transition 数据集，并验证 Episode 边界、数据划分、动作范围和磁盘读取完整性。

## 数据来源

- 随机策略：20 条 Episode，用于提供较分散但任务质量较低的轨迹。
- P 控制策略：20 条 Episode，用于提供能够完成任务的控制轨迹。
- 两种策略逐项使用相同的 20 个环境 seed，因此面对相同目标。
- 环境控制周期为 `0.05 s`，每条 Episode 最多 400 步。

## Transition 结构

每一行包含：

```text
observation[t]
action[t]
reward[t]
next_observation[t]
terminated[t]
truncated[t]
episode_id
step_id
environment_seed
policy_id
split_id
```

`episode_id` 和 `step_id` 用于恢复轨迹顺序；`policy_id` 记录行为来源；`split_id` 防止同一 Episode 跨越训练集和测试集。

## 数据划分

在每种策略内部按完整 Episode 划分：

- train：14 条随机 + 14 条 P 控制 Episode。
- validation：3 条随机 + 3 条 P 控制 Episode。
- test：3 条随机 + 3 条 P 控制 Episode。

划分单位是 Episode，不是 Transition，因此同一轨迹的相邻时间步不会泄漏到不同集合。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_trajectory_dataset.py
PYTHONPATH=src python experiments/020_simulation_dataset/run.py
python -m json.tool outputs/020_simulation_dataset/manifest.json
```

## 输出

- `dataset.npz`：压缩后的纯数值 Transition 数组，读取时禁用 pickle。
- `manifest.json`：字段定义、shape、dtype、数据来源、split 和统计信息。
- `dataset_distributions.png`：不同采集策略产生的目标误差与动作分布。
- `run.log`：采集和验证摘要。

## 完整性检查

- 所有字段具有相同 Transition 数量。
- `next_observation[t]` 与同一 Episode 的 `observation[t+1]` 一致。
- 只有 Episode 最后一行可以 terminated 或 truncated。
- Action 必须位于环境声明范围内。
- 同一 Episode 的 seed、policy 和 split 不得变化。
- 保存后重新加载，所有数组必须逐项一致。
- manifest 保存由数组内容计算的 SHA-256，避免依赖 NPZ 压缩文件的时间元数据。

## 实际结果

- 共采集 40 条 Episode，转换为 `12102` 条 Transition，压缩 NPZ 大小约 `420 KiB`。
- 随机策略：20 条 Episode、`7394` 条 Transition、成功率 `20%`。
- P 控制策略：20 条 Episode、`4708` 条 Transition、成功率 `100%`。
- train、validation、test 分别包含 28、6、6 条完整 Episode，每个 split 都包含两种策略。
- 数据保存后重新读取，全部数组逐项一致。
- 使用相同配置重复采集后，内容 SHA-256 都是 `52004c8c3d23d9a06c16c3c50227851f2b945a02666f8723f66e7523be2b5554`。

两种策略各有 20 条 Episode，但 Transition 数并不相同。随机策略更常运行到 400 步上限，因此若训练时对所有 Transition 等概率采样，随机数据会比 P 控制数据占据更大权重。
