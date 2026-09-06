# ROS 2 Launch

## 快速复习

- Launch 用一个入口描述和启动多个进程，适合复现完整机器人系统的运行方式。
- Python Launch 文件必须提供 `generate_launch_description()`，并返回 `LaunchDescription`。
- `launch_ros.actions.Node` 描述要启动的 ROS 节点进程，不是 `rclpy.node.Node` 基类。
- `DeclareLaunchArgument` 声明 Launch 命令行参数，`LaunchConfiguration` 在执行时取得参数值。
- Launch 参数与 ROS 节点 Parameter 是两个层次，需要在 `Node(parameters=...)` 中显式连接。
- Python 包中的 Launch 文件必须通过 `setup.py` 安装，才能被 `ros2 launch <package> <file>` 找到。

## Launch 解决什么问题

手动运行多个节点需要多个终端、重复命令和人为保持参数一致。Launch 把这些信息集中成一个可执行的系统描述：

```text
LaunchDescription
├── 声明启动参数
├── 启动 position_publisher
└── 启动 position_subscriber
```

Launch 负责进程编排，不替代 Topic、Service、Action 或 Parameter；节点之间仍然通过 ROS 接口通信。

## 当前文件的关键组成

节点启动动作包含：

- `package`：功能包名称。
- `executable`：`setup.py` 注册的可执行入口。
- `name`：运行后的节点名称。
- `output="screen"`：把子进程日志输出到 Launch 终端。
- `parameters`：传递给该节点的 ROS Parameter。

可配置参数的数据流是：

```text
命令行 Launch argument
→ LaunchConfiguration
→ ParameterValue(value_type=float)
→ Node(parameters=...)
→ position_publisher 的 ROS Parameter
```

`ParameterValue` 明确指定浮点类型，避免命令行文本被当成字符串。

## 安装与运行

`setup.py` 使用 `data_files` 把 `launch/*.launch.py` 安装到：

```text
install/point_robot_ros/share/point_robot_ros/launch/
```

修改 Launch 文件或安装规则后应重新构建并加载 overlay：

```bash
colcon build --symlink-install --packages-select point_robot_ros
source install/setup.bash
```

查看参数但不启动节点：

```bash
ros2 launch point_robot_ros point_robot.launch.py --show-args
```

传入参数并启动系统：

```bash
ros2 launch point_robot_ros point_robot.launch.py \
  initial_x:=2.0 \
  velocity_x:=1.0 \
  timer_period:=0.2
```

## 当前项目的实际验证

- `--show-args` 正确显示三个参数、说明和默认值。
- Launch 同时启动 `/position_publisher` 与 `/position_subscriber`。
- `/point_robot/position` 显示 1 个 publisher 和 1 个 subscription。
- 节点参数实际为 `initial_x=2.0`、`velocity_x=1.0`、`timer_period=0.2`。
- 发布者和订阅者显示一致的位置序列。
- `test_position_topic_launch.py` 现在会自动启动发布者，创建临时订阅者，在 5 秒内接收至少 3 条消息，并检查 x 递增、相邻步长约为 0.05。它验证实际 Topic 通信，不只是代码格式。

## 重要边界

- 同一 Topic 可以合法存在多个发布者；若意外启动两套系统，订阅者会收到交错的数据流。
- 节点同名也可能同时存在，使 CLI 查询和日志判断变得混乱。
- 前台运行 Launch 时使用 `Ctrl+C`，让 Launch 统一清理子进程；不要只结束父进程后留下孤立节点。
- 自动通信测试使用受控参数与临时订阅者，不等于验证所有 Launch 分支、硬件或网络场景。具体结构见[最小集成测试](ros2-integration-testing.md)。
