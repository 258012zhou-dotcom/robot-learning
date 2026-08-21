# ROS 2 Point Robot Workspace

这是阶段 2 持续扩展的 ROS 2 Humble 工作空间。当前只包含基础 Python 功能包 `point_robot_ros`，尚未实现节点通信。

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
