# 实验 045：公开 ALOHA 数据的元数据审计

## 目标与原理

实验 044 在**仿真**轨迹中使用 `observation[t]` 对齐 `action[t:t+H]`。真实示范的同一行是否代表同一控制时刻，不能仅凭数组行号判断。训练序列策略前，先核对观察（observation）、动作（action）、时间戳（timestamp）和轨迹边界（Episode）的定义。

本实验只做在线公开资料的**只读审计**，不下载约 1.57 GB 的视频数据，不训练模型，也不操作真机。审计日期：2026-09-27。

## 数据来源与核对方法

- 数据集：[LeRobot `aloha_static_coffee`](https://huggingface.co/datasets/lerobot/aloha_static_coffee)。LeRobot [合并记录 #133](https://github.com/huggingface/lerobot/pull/133) 将 `aloha_static_coffee` 列在真实世界 ALOHA 数据导入清单中；这支持其真机来源，但不等于我们独立核验了采集设备和每帧同步。
- 以仓库当前的 [`meta/info.json`](https://huggingface.co/datasets/lerobot/aloha_static_coffee/blob/main/meta/info.json) 为字段和版本依据，并用[网页数据预览](https://huggingface.co/datasets/lerobot/aloha_static_coffee/viewer/default/train)抽查前几行。数据集卡片内嵌的旧 JSON 写 `v2.0`，而当前 `meta/info.json` 写 `v3.0`；因此不要照抄卡片中的旧文件路径来写读取器。
- 对照 [ACT 原作者的 HDF5 读取代码](https://github.com/tonyzhaozh/act/blob/main/utils.py)了解真实示范的时间偏移风险。该代码不是当前 LeRobot v3 数据集的读取器，不能由它直接推定这个转换后数据集必须偏移一帧。
- 阅读 [ALOHA 原始采集程序](https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/record_episodes.py)、[动作与环境接口](https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/real_env.py)和 [LeRobot 历史 HDF5 转换器](https://github.com/huggingface/lerobot/blob/8e7d697/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py)。这些是代码层面的来源证据，不是对当前全部 50 条 Episode 的逐帧核验。

## 从当前元数据确认的结构

| 项目 | 当前元数据 | 对建模的含义 |
| --- | --- | --- |
| 格式与规模 | LeRobot `v3.0`，50 条 Episode，55,000 帧，1 个任务 | 是数据集规模，不是 50 次成功的闭环评价 |
| 名义频率 | `fps=50`，对应名义帧间隔 `1/50=0.02 s` | 频率字段不能证明所有相机与电机数据的采集延迟为零 |
| 图像 | `cam_high`、`cam_low`、左右腕相机，共 4 路，每帧 `480×640×3` | 可作为视觉观察；尚未抽样检查图像质量或同步 |
| 状态与动作 | `observation.state`、`observation.effort`、`action` 都是 14 维；名字按左臂 7 维、右臂 7 维排列，含 gripper | 元数据给出通道名字，未说明动作是位置目标、增量还是速度，也未说明所有通道的单位 |
| 时间与边界 | 有 `episode_index`、`frame_index`、`timestamp`、`next.done` | 构造片段时必须限定在同一 Episode；还需检查每条轨迹末尾和时间连续性 |
| 数据划分 | 元数据只有 `train: 0:50` | 不存在可直接引用的官方验证集/测试集；后续须按 Episode 自行划分并固定种子 |

网页预览的 Episode 0 前三行依次为 `frame_index=0,1,2`，`timestamp=0,0.02,0.04` 秒。这只验证了**预览样例**与 50 fps 一致，未统计全量轨迹的丢帧、重复帧或非单调时间戳。

## 本次进一步核对：一行数据对应什么

先分清三个时刻：`observation[t]` 是机器人/相机被读取的时刻；`action[t]` 是主臂位置被读出并形成命令的时刻；命令真正让从臂运动，还会经过通信和执行器响应。即使软件将前两者写在同一行，也不代表传感器与执行器零延迟。

原始 ALOHA 采集程序按下面的顺序组织一轮（简写）：

```text
已有观察 obs[t] → 读取主臂形成 action[t] → env.step(action[t]) → 得到 obs[t+1]
保存的一行：obs[t] 与 action[t]
```

`real_env.py` 中手臂每侧前 6 维来自主臂关节位置，并作为从臂的**绝对关节位置目标**发送；每侧第 7 维是归一化的夹爪开合命令。原始程序的归一化约定以 `0` 表示闭、`1` 表示开，见[环境接口](https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/real_env.py)和[常量定义](https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/constants.py)。这说明原始程序**不是**把 14 维动作统一当成速度或增量；关节位置通常按弧度表示，夹爪归一化值则不是弧度。对这份转换后数据，仍需核对采集脚本版本和具体数值，不能仅凭通道名断言全部单位已独立验证。

LeRobot 历史转换器将 HDF5 的 `/observations/qpos` 与 `/action` **按原数组索引**写为 `observation.state` 和 `action`，没有在该步显式移动一帧；它还用 `frame_index / fps` 构造 `timestamp`。所以连续的 `0、0.02、0.04` 更直接证明的是**名义索引时间**，不是相机曝光、主臂读取或从臂执行的实测时间。当前仓库已升级到 v3.0；这些历史代码提供来源线索，不代替对 v3 文件逐帧检查。

## 关键边界与下一步

ACT 原作者的读取器对模拟数据从 `action[start_ts]` 起取片段，对真实 HDF5 示范从 `action[max(0,start_ts-1)]` 起取，并标注是为了让时间步更对齐。这是**原始 ACT 数据处理中的做法**，不是当前 LeRobot 数据的已证实对齐规则。不能不经检查就将实验 044 的 `action[t]` 起点直接用于这份数据，也不能机械地减一帧。

下一步先用**教学用合成序列**比较 `action[t]` 和 `action[t-1]` 两种标签会如何改变片段起点；这只帮助理解对齐，不是从真实数据估计延迟。若以后下载小范围真实样本，再核对原始采集版本、转换链和实际时间/运动关系；确认后才考虑在真实数据上构造训练片段。闭环成功率仍需独立评价，离线误差不能代替。
