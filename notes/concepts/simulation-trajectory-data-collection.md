# 仿真轨迹数据采集

## 快速复习

- 机器人学习数据是有时间顺序的轨迹，不是彼此独立的普通表格行。
- 一条 Transition 通常包含 `(observation, action, reward, next_observation, terminated, truncated)`。
- 执行 `T` 个 Action 会得到 `T+1` 个 Observation，并形成 `T` 条 Transition。
- train、validation、test 应按完整 Episode 划分，避免相邻时间步泄漏。
- 必须记录 Episode、step、seed、策略来源、控制周期和结束原因。
- Episode 数量相等不代表 Transition 数量或训练权重相等。
- 保存成功不代表数据正确；必须重新读取并验证 shape、边界、连续性和元数据。
- 数据分布由采集策略决定，随机数据、专家数据和已训练策略数据不能混为一谈。

## 从 Episode 到 Transition

一次环境交互是：

```text
observation[t]
    ├── policy → action[t]
    └── environment.step(action[t])
            ↓
reward[t], observation[t+1], terminated[t], truncated[t]
```

因此第 `t` 条 Transition 是：

```text
(当前观测, 当前动作, 本步奖励, 下一观测, 是否任务终止, 是否外部截断)
```

如果一个 Episode 执行 `T` 次动作：

```text
observations.shape      = (T + 1, observation_dimension)
actions.shape           = (T, action_dimension)
rewards.shape           = (T,)
next_observations[t]    = observations[t + 1]
```

初始 Observation 没有对应的前一个 Action，不能为了数组等长而随意删除。转换为 Transition 表时，应使用 `observations[:-1]` 作为当前状态，使用 `observations[1:]` 作为下一状态。

## 为什么需要 Episode ID 和 Step ID

把所有 Transition 拼接成一个大数组后，原来的轨迹边界会消失。因此每一行还要保存：

- `episode_id`：这条 Transition 属于哪条轨迹。
- `step_id`：它在该轨迹中的时间位置。

它们用于：

- 恢复完整轨迹；
- 计算每条 Episode 的成功率和回报；
- 构造历史窗口或 Action Chunk；
- 防止模型跨 Episode 读取并不存在的下一状态；
- 检查 step 是否连续、是否重复或丢失。

只有每条 Episode 的最后一条 Transition 可以出现 `terminated=True` 或 `truncated=True`。

## terminated 与 truncated 必须分开保存

- `terminated`：任务自身结束，例如稳定到达目标。
- `truncated`：外部限制结束，例如达到最大步数。

二者都代表当前 Episode 不再继续，但语义不同。强化学习计算 bootstrap target 时，时间截断不一定等同于真正终止。只保存一个模糊的 `done` 会丢失这部分信息。

二者**允许同时为真**：例如任务成功时，外层 `TimeLimit` 恰好达到步数上限。完整 Episode 的最后一行至少一个结束标记为真，而不是恰好一个；两个字段都必须是布尔数组。项目测试用真实 `TimeLimit` 制造该边界，验证转换、NPZ 保存与加载都保留两个 `True`，同时继续拒绝尚未结束的 Episode。

## 按 Episode 划分数据集

相邻机器人状态通常高度相似。如果先打散所有 Transition，再随机划分：

```text
同一轨迹的 step 50 → train
同一轨迹的 step 51 → test
```

测试集几乎包含训练样本的邻居，会导致数据泄漏和虚高结果。

正确流程是：

```text
先确定完整 Episode 属于哪个 split
        ↓
再把 split_id 写入该 Episode 的所有 Transition
```

实验 020 还按采集策略分层，使随机策略和 P 控制策略都出现在 train、validation 和 test 中。分层只能保证策略类别存在，不能自动保证每个 split 的目标难度和成功率完全相同；小数据集仍可能有统计波动。

## Episode 分布与 Transition 分布

实验 020 为两种策略各采集 20 条 Episode：

| 策略 | Episode | Transition | 成功率 |
| --- | ---: | ---: | ---: |
| Random | 20 | 7394 | 20% |
| P control | 20 | 4708 | 100% |

随机策略经常运行到 400 步上限，P 控制策略通常提前成功，所以随机策略产生更多 Transition。

如果训练时均匀抽取所有 Transition，随机策略数据约占：

```text
随机策略数据占比 = 7394 / 12102 ≈ 61.1%
```

这与“每种策略各占一半 Episode”不是同一个分布。可选处理方法包括：

- 均匀采样 Transition：长 Episode 权重更大。
- 先均匀采样 Episode，再采样其中的 step：每条轨迹权重接近。
- 按策略或成功标签重新加权。
- 明确只使用某类策略数据，例如行为克隆只使用专家示范。

选择哪一种取决于训练目标，不能默认其中一种永远正确。

## 采集策略决定数据分布

随机策略产生近似覆盖整个动作范围的数据，但大量状态与任务成功无关。P 控制策略集中在朝目标运动和目标附近的状态，动作更多分布在零附近。

这会影响模型学到什么：

- 只使用成功控制轨迹，模型可能不知道偏离很远后如何恢复。
- 只使用随机轨迹，数据覆盖广，但正确行为信号稀少。
- 混合数据可以增加覆盖，但必须保留策略来源和采样权重。

随机策略在实验 020 的成功率为 20%，实验 018 为 7.5%。两次使用的策略随机 seed 和样本数量不同，这种差异不能解释为策略能力提升。评估随机系统时需要更多 seed、置信区间或重复实验。

## 数据字段与元数据

实验 020 的每条 Transition 保存：

```text
observations
actions
rewards
next_observations
terminated
truncated
episode_ids
step_ids
environment_seeds
policy_ids
split_ids
```

manifest 另外保存：

- 字段 shape 和 dtype；
- Observation 与 Action 的语义；
- Action 范围；
- 控制周期；
- 数据来源模型；
- 策略和 split 的编号映射；
- Episode、Transition 和成功率统计；
- 每个 split 包含的 Episode ID；
- 数据内容哈希。

数组本身和 manifest 缺一不可：前者提供训练数据，后者提供解释和复现依据。

## commanded Action 与 executed Action

真实机器人中的策略指令可能经过限幅、安全监督、底层控制器和通信系统：

```text
policy command
    ↓ 限幅、滤波、安全检查
executed command
    ↓ 执行器与动力学
observed motion
```

可靠数据集应尽量区分策略请求的 `commanded_action` 和实际执行的 `executed_action`。实验 020 的策略已经产生合法动作，环境执行的值与保存值相同，因此暂时只保存一个 Action 字段。

## NPZ、JSON 与安全读取

当前低维数据使用：

- `dataset.npz`：压缩保存 NumPy 数组。
- `manifest.json`：保存可读的 schema、来源和统计。

读取 NPZ 时使用 `allow_pickle=False`，确保文件只包含普通数值数组，不执行 Python pickle 对象。随着数据扩展到大量图像、点云和长序列，再考虑 HDF5、Zarr、Parquet、RLDS 或 LeRobot Dataset。

## 数据完整性验证

实验 020 自动检查：

- 所有字段的第一维等于 Transition 数量；
- Observation 与 next Observation shape 一致；
- 数值字段没有 `NaN` 或无穷值；
- Action 没有超出声明范围；
- 每条 Episode 的 step 从零连续递增；
- `next_observation[t]` 等于下一行的 `observation[t+1]`；
- seed、policy 和 split 在同一 Episode 内保持不变；
- 只有最后一行包含结束标记；
- NPZ 保存再读取后所有数组逐项一致。

文件存在、大小正常或程序没有报错，都不能替代这些检查。

## 可复现性与内容哈希

实验将字段名、dtype、shape 和数组字节依次送入 SHA-256。相同配置重复采集两次，内容哈希均为：

```text
52004c8c3d23d9a06c16c3c50227851f2b945a02666f8723f66e7523be2b5554
```

这里计算的是解压后的数组内容，而不是整个 NPZ 文件字节，因此不会受到压缩文件内部时间元数据影响。哈希相同可以证明内容相同，但不能证明数据在科学意义上正确；物理语义和采集设计仍需要单独验证。

## 图像和多传感器数据的额外问题

低维同步状态可以直接按 step 保存。真实机器人数据还需要处理：

- 摄像头、关节状态和控制指令频率不同；
- 传感器时间戳来自不同设备时钟；
- 图像可能丢帧、延迟或压缩；
- 一个策略 Observation 可能需要最近帧、插值状态或历史窗口；
- 原始数据、预处理输入和模型特征不能混淆。

后续接触真实机器人数据集时，需要把时间同步和数据版本作为数据契约的一部分。

## 当前边界与下一步

- 当前数据只有一维仿真状态和单维 Action，不代表真实机器人数据复杂度。
- 数据规模为 40 条 Episode，只用于验证管线。
- 尚未把数据用于行为克隆或离线强化学习训练。
- 尚未加入动力学、传感器和控制参数随机化。
- 下一模块将学习领域随机化：明确哪些环境参数可以变化，并严格区分训练随机化分布与固定评估分布。
