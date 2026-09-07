# 实验 024：Piper 机械臂盘点与一次只读反馈

## 这次只做什么

先确认机械臂和软件环境，再读取一次状态。暂时不让机械臂运动。

程序只读取：

- 6 个关节角；
- 夹爪开度；
- 6 个电机是否使能；
- 机械臂状态码；
- 反馈频率和 Piper SDK 版本。

完整流程只有：

```text
检查配置 → 连接 can0 → 等待反馈 → 读取一次 → 单位转换 → 保存 → 断开
```

## 从现有资料确认的信息

- 铭牌已确认品牌为 AGILEX / 松灵机器人，型号为 PIPER。
- 完整设备序列号只离线保留，不写入公开 GitHub 仓库。
- CAN 接口为 `can0`，历史设置的 bitrate 为 1,000,000。
- `piper_sdk 0.6.1` 已安装在 `lerobot`、`piper_lerobot` 和个人 `xvla` 环境中。
- 历史 ROS 2 节点是 `piper_single_ctrl`，设置 `auto_enable:=false` 后可发布 `/joint_states_feedback`、`/arm_status` 和 `/end_pose`。
- 当前 SSH 环境没有发现可直接使用的 `ros2` 命令或 `piper` ROS 2 包，所以本实验先使用已有 SDK。
- RealSense 历史启动命令为 `ros2 launch realsense2_camera rs_launch.py`；相机不属于本次只读关节实验。

## 资料记录的 ROS 2 启动顺序

下面用于理解旧系统，不是本次实验要执行的命令：

```bash
# 1. 将 can0 设置为 1 Mbps
bash can_activate.sh can0 1000000

# 2. 启动 Piper 驱动，但不自动使能
ros2 run piper piper_single_ctrl --ros-args \
  -p can_port:=can0 \
  -p auto_enable:=false \
  -p gripper_exist:=true \
  -p gripper_val_mutiple:=2

# 3. 读取驱动发布的关节反馈
ros2 topic echo /joint_states_feedback
```

当前电脑尚未定位到 `can_activate.sh`、ROS 2 环境和 `piper` 包，因此不能直接照抄运行。即使 `auto_enable=false`，正式启动前仍要检查源码和急停方式。

## 哪些入口暂时不能使用

下面这些会进入控制链路，本实验不运行：

- `/pos_cmd`：末端位姿和夹爪命令；
- `/joint_ctrl_single`：关节命令；
- `/enable_flag`：使能控制；
- `JointCtrl`、`MotionCtrl`、`GripperCtrl`：SDK 控制函数；
- ACT 推理、校准和 `safe_disconnect()`：可能使能或移动机械臂。

## 真机退出策略

第一轮真机实验采用以下硬约束：

- 正常结束时停止并保持当前位置，不自动归零。
- 不自动调用 `DisableArm()` 或 `ResetPiper()`，避免机械臂失去力矩后下坠。
- 不使用当前 ACT 的 `safe_disconnect()`，因为它会先移动再失能。
- 机械臂得到机械支撑前，不执行失能或断电测试。
- 实体急停已经找到，但按下后的保持或掉电行为仍需在安全支撑条件下确认。

## 为什么还要换算单位

SDK 的关节反馈不是直接的弧度：

```text
角度 = raw × 0.001°
弧度 = raw × 0.001 × π / 180
夹爪开度 = raw × 0.000001 m
```

项目同时保存 raw、度和弧度，方便发现单位错误。ACT 适配代码中的换算曾被注释，因此在控制机械臂前必须单独核对。

## 当前可以安全运行的检查

这一步只读取 JSON 配置，不导入 SDK、不连接 CAN：

```bash
cd /media/zhao/F/zxd/robot-learning
PYTHONPATH=src /media/zhao/F/envs/xvla/bin/python \
  experiments/024_arm_inventory/run.py
```

看到“未连接 CAN，也没有发送任何命令”即为通过。

## 下一步：一次只读反馈

先完成三件事：

1. 确认配置中的型号为 `PIPER`，且公开文件不包含完整序列号。
2. 确认机械臂静止，周围无人，急停或断电方式可用。
3. 确认没有 ACT、ROS 2 或其他 Piper 程序占用 `can0`。

满足后才执行：

```bash
PYTHONPATH=src /media/zhao/F/envs/xvla/bin/python \
  experiments/024_arm_inventory/run.py --read-once
```

结果写入 `outputs/024_arm_inventory/results.json`，默认不会提交 Git。

## 这一步怎样算通过

- 程序正常退出，没有机械臂运动；
- `joint_feedback_hz` 大于 0；
- 六个关节角和当前姿态看起来一致；
- 夹爪开度在合理范围内；
- `status_code` 的含义得到确认；
- 结果中的 `enable_command_sent` 和 `motion_command_sent` 均为 `false`。

通过后再增加速度、温度和详细故障位；不在第一份代码中一次塞完。

## 参考

- [AgileX Piper SDK](https://github.com/agilexrobotics/piper_sdk)
- [Piper SDK V2 interface](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD)
