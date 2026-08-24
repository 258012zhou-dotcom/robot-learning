# ROS 2 Point Robot Workspace

这是阶段 2 持续扩展的 ROS 2 Humble 工作空间。当前包含节点包 `point_robot_ros` 和接口包 `point_robot_interfaces`，已经实现 Topic、Service 和 Action 通信。

## 环境

ROS 2 使用 Ubuntu 系统 Python，与根项目的 Conda 环境分开。每个新终端先在项目根目录执行：

```bash
source scripts/activate_ros2_project.sh
```

该脚本加载 ROS 2 Humble 和当前工作空间，并统一设置 Cyclone DDS 与 Domain 31。不要把 ROS 环境全局写入普通 Conda 终端，以免 ROS 的 Python 路径和 pytest 插件污染机器学习环境。

当前还启用了用户服务 `ros2-cli-daemon-domain31.service`，用于持续维护 Domain 31 的 ROS 2 CLI daemon。服务定义保存在 [`systemd/`](systemd/)；安装、检查和撤销方法见 [DDS 与 daemon 故障记录](../../notes/troubleshooting/ros2-dds-wsl-tun.md)。

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

- `point_robot_ros` 的 Flake8 和 PEP 257 通过。
- 版权头测试按模板默认跳过。
- `point_robot_interfaces` 的 CMake lint 和 XML schema 检查通过。
- `test_position_topic_launch.py` 会自动启动位置发布者，并由临时测试节点通过 DDS 订阅位置 Topic。
- 使用独立的 `ROS_DOMAIN_ID=132` 运行后，当前汇总为 8 tests、0 errors、0 failures、1 skipped。

其中代码规范测试不代表通信功能正确；新增的集成测试验证了位置 Topic 能收到至少三条消息、`y` 保持为零、`x` 递增且相邻步长符合启动参数。Service、Action 和完整 Launch 系统目前仍以手动运行验证为主。

为了避免日常 Domain 31 中的节点干扰，集成测试使用临时 DDS Domain：

```bash
ROS_DOMAIN_ID=132 colcon test \
  --packages-select point_robot_ros \
  --event-handlers console_direct+
colcon test-result --verbose
```

测试的结构和证据边界见 [ROS 2 最小集成测试笔记](../../notes/concepts/ros2-integration-testing.md)。

## Parameter

位置发布者声明三个参数：

- `initial_x`：启动位置，只能在启动时设置。
- `timer_period`：发布周期，只能在启动时设置，且必须大于零。
- `velocity_x`：x 方向速度，支持运行时动态修改。

启动时覆盖默认值：

```bash
ros2 run point_robot_ros position_publisher \
  --ros-args \
  -p initial_x:=0.0 \
  -p velocity_x:=1.0 \
  -p timer_period:=0.2
```

运行时修改速度：

```bash
ros2 param set /position_publisher velocity_x 2.0
```

实际验证中，速度为 `1.0`、周期为 `0.2 s` 时位置每次增加 `0.20`；动态修改速度为 `2.0` 后，每次增加 `0.40`。修改 `timer_period` 会被回调拒绝并返回 `timer_period is startup-only`。

参数声明、读取、验证和动态回调的关系见 [ROS 2 Parameter 笔记](../../notes/concepts/ros2-parameters.md)。

## Launch

`point_robot.launch.py` 同时启动位置发布者、订阅者、动态 TF 广播节点和 `robot_state_publisher`，并把 Launch arguments 转换为发布者的节点参数、把已安装的 URDF 加载为 `robot_description`。

查看可用参数：

```bash
ros2 launch point_robot_ros point_robot.launch.py --show-args
```

使用自定义参数运行：

```bash
ros2 launch point_robot_ros point_robot.launch.py \
  initial_x:=2.0 \
  velocity_x:=1.0 \
  timer_period:=0.2
```

实际验证中，Launch 同时创建 `/position_publisher`、`/position_subscriber`、`/position_tf_broadcaster` 和 `/robot_state_publisher`。位置 Topic 显示 1 个发布者和 2 个订阅者，三个位置发布参数均与命令行输入一致。前台运行时使用 `Ctrl+C`，由 Launch 统一停止四个子进程。

Launch 文件结构、参数传递和进程边界见 [ROS 2 Launch 笔记](../../notes/concepts/ros2-launch.md)。

## rosbag 记录与回放

位置 Topic 的实验数据保存在根项目的 `data/local/rosbags/`，该目录已被 Git 忽略。

```bash
ros2 bag record \
  /point_robot/position \
  -o data/local/rosbags/point_robot_position

ros2 bag info data/local/rosbags/point_robot_position
ros2 bag play data/local/rosbags/point_robot_position
```

实际记录结果为 16.60 秒、167 条 `geometry_msgs/msg/Point` 消息，平均频率约为 10 Hz，SQLite3 数据文件大小为 33.0 KiB。停止原发布者后，回放进程重新发布 `/point_robot/position`；订阅者在回放前等待、回放期间连续接收、回放结束后停止接收。

这证明选定 Topic 的消息和时间关系可以离线复现，但不代表发布节点的内部状态、Service、Action 或整个系统执行过程都已保存。详细说明见 [ROS 2 rosbag 笔记](../../notes/concepts/ros2-rosbag.md)。

## TF2 坐标系统

`position_tf_broadcaster` 订阅 `/point_robot/position`，把位置消息转换为动态变换：

```text
world → base_link
```

该变换发布到 `/tf`，时间戳来自节点时钟，平移取自 `Point` 消息，当前旋转使用单位四元数。节点已经加入 `point_robot.launch.py`。

```bash
ros2 launch point_robot_ros point_robot.launch.py
ros2 run tf2_ros tf2_echo world base_link
```

实际验证包括：

- 静态建立 `world → base_link → camera_link` 两级坐标树。
- TF2 正确组合得到 `world → camera_link` 平移 `[1.2, 0.5, 0.3]`。
- `camera_link` 绕 z 轴旋转 90° 后，四元数约为 `[0, 0, 0.707, 0.707]`。
- 动态变换在约 1 秒内从 `x=35.3` 更新到 `x=35.8`，符合默认速度 `0.5 m/s`。
- `view_frames` 显示 `world → base_link`，平均发布频率约为 `10.196 Hz`，缓存跨度约为 `5.1 s`。

TF2 的坐标树、变换公式、时间语义与当前边界见 [ROS 2 TF2 笔记](../../notes/concepts/ros2-tf2.md)。

## URDF 机器人模型

`urdf/point_robot.urdf` 描述一个蓝色箱体底座和固定安装的摄像头：

```text
base_link
└── camera_joint (fixed)
    └── camera_link
```

底座尺寸为 `0.6 × 0.4 × 0.2 m`，摄像头尺寸为 `0.12 × 0.08 × 0.08 m`。`camera_joint` 将摄像头放在底座前方 `0.25 m`、上方 `0.28 m`。两个 Link 都包含 visual、collision、mass 和 inertia。

URDF 由 `setup.py` 安装到功能包共享目录。Launch 从安装目录读取文件，并通过标准参数 `robot_description` 交给 `robot_state_publisher`。

```bash
check_urdf src/point_robot_ros/urdf/point_robot.urdf
ros2 launch point_robot_ros point_robot.launch.py
ros2 run tf2_ros tf2_echo base_link camera_link
ros2 run tf2_ros tf2_echo world camera_link
```

实际验证结果：

- `check_urdf` 成功解析，根 Link 为 `base_link`，子 Link 为 `camera_link`。
- 固定变换 `base_link → camera_link` 为 `[0.25, 0, 0.28]`，时间显示为 `0.0`。
- TF2 能组合动态与静态关系，得到连续更新的 `world → camera_link`。
- `/tf` 发布端来自动态广播节点和 `robot_state_publisher`，`/tf_static` 由 `robot_state_publisher` 发布固定关节。
- 修改后测试汇总为 8 tests、0 errors、0 failures、1 skipped。

模型结构和发布流程见 [ROS 2 URDF 笔记](../../notes/concepts/ros2-urdf.md)。模型外观尚待 RViz 实际检查。

## RViz 可视化

保存的配置位于 `rviz/point_robot.rviz`，固定坐标系为 `world`，并启用了 Grid、RobotModel 和 TF。配置由 `setup.py` 安装到功能包共享目录，可通过 Launch 参数选择是否启动 GUI：

```bash
ros2 launch point_robot_ros point_robot.launch.py \
  velocity_x:=0.0 \
  use_rviz:=true
```

`use_rviz` 默认为 `false`，因此后台运行和自动测试不会强制打开图形界面。

实际验证结果：

- RViz 正确显示蓝色 `base_link` 和深灰色 `camera_link`。
- TF 显示为 `world → base_link → camera_link`，坐标轴与名称可见。
- 运行时把 `velocity_x` 从 `0.0` 改为 `0.1` 后，两个 Link 保持固定相对关系并沿 world 的 x 轴一起移动。
- 再把速度设为 `0.0` 后模型停止；调用 Reset Service 后模型返回原点附近。
- 保存配置后，`use_rviz:=true` 能自动恢复 Fixed Frame、RobotModel、TF 和观察视角。
- 修改后测试汇总保持 8 tests、0 errors、0 failures、1 skipped。

RViz 的显示数据流和诊断边界见 [ROS 2 RViz 笔记](../../notes/concepts/ros2-rviz.md)。RViz 是可视化工具，不负责物理仿真、碰撞响应或机器人控制。

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

## MoveToPosition Action

自定义接口由独立包 `point_robot_interfaces` 提供：

```text
Goal
  target_x
  max_speed
---
Result
  success
  final_x
  message
---
Feedback
  current_x
  remaining_distance
```

运行 Action Server：

```bash
ros2 run point_robot_ros move_action_server
```

运行位置订阅者：

```bash
ros2 run point_robot_ros position_subscriber
```

可使用 CLI Client 发送目标：

```bash
ros2 action send_goal \
  /point_robot/move_to_position \
  point_robot_interfaces/action/MoveToPosition \
  "{target_x: 1.0, max_speed: 0.2}" \
  --feedback
```

也可以运行 Python Client：

```bash
ros2 run point_robot_ros move_action_client
```

实际验证结果：

- Goal 被接受。
- Action Feedback 与位置 Topic 同步增长。
- 到达目标后返回 `success=True`、正确的 `final_x` 和 `Target reached`。
- Goal 最终状态为 `SUCCEEDED`。
- 客户端取消后，服务器停止运动并返回取消位置，Goal 最终状态为 `CANCELED`。
- 一个 Goal 执行期间发送第二个 Goal，第二个 Goal 被拒绝，避免两个任务同时修改位置。
- 服务器可以通过 `Ctrl+C` 干净退出。

取消测试可以发送一个耗时目标，在持续收到 Feedback 时按 `Ctrl+C`：

```bash
ros2 action send_goal \
  /point_robot/move_to_position \
  point_robot_interfaces/action/MoveToPosition \
  "{target_x: 10.0, max_speed: 0.2}" \
  --feedback
```

当前 Action Server 使用两线程 executor 和可重入回调组，使执行目标期间仍能处理取消请求。`Lock` 与活动目标标记实现“同一时间只执行一个 Goal”的策略；第二个并发目标会得到 `Goal was rejected`。

运行 Action Server 时不要同时运行旧的 `position_publisher`，否则 `/point_robot/position` 会有两个发布者。当前系统仍是软件模拟的点机器人，不代表真实电机控制结果。
