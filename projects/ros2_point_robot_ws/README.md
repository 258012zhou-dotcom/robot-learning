# ROS 2 Point Robot Workspace

这是阶段 2 持续扩展的 ROS 2 Humble 工作空间。当前包含 Python 功能包 `point_robot_ros`，已经实现基于 Topic 的位置发布与订阅通信。

## 环境

ROS 2 使用 Ubuntu 系统 Python，与根项目的 Conda 环境分开。每个新终端先执行：

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=31
```

当前 WSL/TUN 环境使用 Cyclone DDS；原因见 [DDS 故障记录](../../notes/troubleshooting/ros2-dds-wsl-tun.md)。

## 结构

```text
ros2_point_robot_ws/
├── src/       # 功能包源码，提交 Git
├── build/     # 构建中间文件，不提交
├── install/   # 安装结果和环境脚本，不提交
└── log/       # 构建及测试日志，不提交
```

## 构建与发现

```bash
cd ~/AI_Project/robot-learning/projects/ros2_point_robot_ws
colcon build --symlink-install
source install/setup.bash
ros2 pkg prefix point_robot_ros
```

`source install/setup.bash` 只影响当前终端。删除工作空间后，旧终端仍可能保存失效的 `AMENT_PREFIX_PATH`，此时应打开新终端并重新加载环境。

## 测试

```bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

当前实际结果：

- 3 个规范测试被收集。
- Flake8 和 PEP 257 通过。
- 版权头测试按模板默认跳过。
- 0 errors，0 failures，1 skipped。

这些测试只验证代码规范，不代表 Node 或 Topic 功能已经正确。

## Topic 通信

终端 A 运行位置发布者：

```bash
ros2 run point_robot_ros position_publisher
```

终端 B 运行位置订阅者：

```bash
ros2 run point_robot_ros position_subscriber
```

两个终端必须使用相同的 `RMW_IMPLEMENTATION` 和 `ROS_DOMAIN_ID`。当前 Topic 接口为：

```text
/point_robot/position [geometry_msgs/msg/Point]
```

实际运行验证结果：

- 发布频率约为 `10 Hz`。
- 订阅者连续收到位置消息，`x` 每次增加 `0.05`，`y` 保持为 `0.0`。
- `ros2 topic info /point_robot/position` 显示 1 个 publisher 和 1 个 subscription。
- 新订阅者从加入后的消息开始接收，不保证取得连接前的历史数据。

## Reset Service

位置发布节点同时提供重置服务：

```text
/point_robot/reset [std_srvs/srv/Trigger]
```

调用命令：

```bash
ros2 service call \
  /point_robot/reset \
  std_srvs/srv/Trigger \
  "{}"
```

实际运行验证结果：

- 服务返回 `success=True` 和 `Position reset to x=0.0`。
- 调用前发布位置为 `25.10`。
- 服务回调执行后，下一条位置消息变为 `0.00`。
- 定时器没有停止，位置随后继续按 `0.05` 递增。

因此该验证不仅检查了响应内容，也通过 Topic 输出确认了节点内部状态确实被重置。

Python 客户端：

```bash
ros2 run point_robot_ros reset_client
```

客户端会等待服务、异步发送空的 `Trigger` 请求、等待 Future 完成并输出响应，然后正常退出。实际验证输出为：

```text
Reset response: success=True message=Position reset to x=0.0
```
