# ROS 2 机器人安全基础

## 快速复习

- 安全层应位于控制器与执行器之间，不能只依赖 PID 自己的输出限幅。
- 安全默认值是零输出：启动时无指令、指令超时、输入非法或急停激活时都应停止。
- 限幅（Saturation）约束单条指令；Watchdog 处理控制指令断流；Emergency Stop 覆盖其他控制意图。
- 解除急停不应恢复旧指令，必须等待新的有效指令。
- 软件急停只能覆盖软件链路中的部分故障，真实机器人仍需要独立硬件急停、驱动器使能和安全操作流程。

## 安全层的位置

项目采用下面的职责分离：

```text
目标或策略
    ↓
控制器：计算期望速度
    ↓ cmd_vel_raw
安全监督器：限幅、Watchdog、急停
    ↓ cmd_vel_safe
执行器或软件运动模型
```

PID 的 `max_command` 能约束正常计算结果，但控制节点可能配置错误、失去通信或产生非有限值。独立安全层把“如何完成任务”和“当前是否允许执行”分开，使不同控制器共享同一组最低安全约束。

## 三项基础机制

### 指令限幅

对一维速度 `u` 和上限 `u_max`：

```text
u_safe = clip(u, -u_max, u_max)
```

限幅防止单条指令超过允许范围，但不能限制速度变化率，也不能判断机器人是否接近障碍物。真实系统通常还需要加速度、关节位置、力矩、工作空间和碰撞限制。

### Watchdog 超时停车

安全监督器保存最后一条有效指令的单调时间戳。如果：

```text
current_time - last_command_time > command_timeout
```

则输出归零并丢弃旧指令。项目使用 `time.monotonic()`，因为 Watchdog 关心经过的真实时长，不应受系统时间校准、时区或 rosbag 仿真时间跳变影响。

### 软件急停

急停激活时，监督器立即清空命令和时间戳，并持续输出零。急停期间收到的新指令会被丢弃。解除急停只改变安全状态，不恢复旧命令；之后必须收到一条新的有效指令。

项目用 `std_srvs/srv/SetBool` 表示软件急停状态：

```text
data: true  → 激活急停
data: false → 解除急停，等待新指令
```

这是一种教学接口。真实系统的急停通常要求物理按钮、硬接线安全回路、驱动器断使能、锁存状态和人工复位，不能只依赖 ROS Service。

## 当前 ROS 2 接口

```text
/point_robot/cmd_vel_raw  [geometry_msgs/msg/Twist]
/point_robot/cmd_vel_safe [geometry_msgs/msg/Twist]
/point_robot/set_emergency_stop [std_srvs/srv/SetBool]
```

`safety_node` 当前只处理 `Twist.linear.x`，其他速度分量保持为零。参数包括：

- `max_linear_speed`：允许的最大线速度绝对值。
- `command_timeout`：没有新指令后允许等待的最长时间。
- `publish_period`：安全输出的发布周期。

总 Launch 默认不启动安全节点；设置 `use_safety:=true` 后启用，避免在不需要速度链路的感知实验中增加无关节点。

## 验证证据

手动 ROS 验证已确认：

- 原始速度 `2.0` 在最大速度 `1.0` 下输出为 `1.0`。
- 单次指令停止发送约 `0.5 s` 后，输出自动变为 `0.0`。
- 连续发送 `0.6` 时激活急停，安全输出立即变为 `0.0`。
- 停止原始发布者并解除急停后，输出仍为 `0.0`；发送新指令后才恢复。
- 总 Launch 使用 `use_safety:=true` 时能发现 `/safety_node` 和急停 Service。

自动验证包括 9 个纯 Python 单元测试和 1 个 ROS Launch 集成测试。完整工作空间结果为 65 tests、0 errors、0 failures、1 skipped。

## 当前边界

本轮完成代表能够解释并验证软件级限幅、Watchdog 和急停恢复逻辑，不代表已经完成真实机器人功能安全。尚未覆盖：

- 硬件急停、驱动器使能与安全继电器。
- 加速度、关节位置、力矩和工作空间限制。
- 碰撞检测、障碍物停车和传感器健康监测。
- 多节点生命周期、故障状态机和冗余控制。
- ISO 10218、ISO/TS 15066 等机器人安全标准的工程实施。
