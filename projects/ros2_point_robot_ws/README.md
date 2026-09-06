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
- `point_robot_interfaces` 的 CMake lint 和离线 XML 有效性检查通过。
- `test_position_topic_launch.py` 会自动启动位置发布者，并由临时测试节点通过 DDS 订阅位置 Topic。
- `test_safety_node_launch.py` 会通过真实 ROS Topic 验证速度限幅和断流超时停车。
- `test_safe_motion_launch.py` 运行安全模式的总 Launch，验证安全输出确实驱动模拟位置，非法输入、急停与断流使位置停止。
- 2026-09-05 使用独立的 `ROS_DOMAIN_ID=132` 和本机通信运行后，完整测试汇总为 70 tests、0 errors、0 failures、1 skipped；其中包含运动学、控制、编码器、里程计、模拟传感器、状态估计和安全监督测试。

其中代码规范测试不代表通信功能正确；三个集成测试覆盖位置 Topic、安全 Topic 和总 Launch 中的安全运动链路。其他 Service、Action 与可视化功能仍包含手动运行验证，不能把测试通过解释为全部机器人行为已自动覆盖。两个 `package.xml` 不再引用在线 Schema，避免断网时把有效 XML 误报为测试失败。

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

`point_robot.launch.py` 同时管理位置通信、动态 TF、`robot_state_publisher`、关节状态发布器和可选 RViz，并把已安装的 URDF 加载为 `robot_description`。

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

`joint_control_mode` 决定 `/joint_states` 的唯一来源，只允许三种取值：`publisher` 使用无界面的默认发布器，`gui` 使用滑块界面，`pid` 使用摄像头 PID 控制节点。单一模式参数从结构上避免多个节点同时控制 `camera_joint`。

例如启动 PID 控制和 RViz：

```bash
ros2 launch point_robot_ros point_robot.launch.py \
  joint_control_mode:=pid \
  target_yaw:=0.8 \
  use_rviz:=true
```

前台运行时使用 `Ctrl+C`，由 Launch 统一清理子进程。

## PID 基础控制

`control.py` 包含限幅 P 控制器、有状态 PID 控制器，以及一阶和二阶关节仿真。测试分别验证控制方向、速度限幅、离散稳定性、积分累计与限幅、稳态误差消除，以及 D 项对惯性关节的阻尼作用。

`camera_pid_controller` 把 PID 接入 ROS 2：节点读取目标角和启动增益，生成受限速度命令，更新软件模拟的 `camera_joint`，并发布 `/joint_states`。这仍是软件模型，不是对真实电机的控制。

```bash
ros2 launch point_robot_ros point_robot.launch.py \
  joint_control_mode:=pid \
  target_yaw:=0.8 \
  kp:=2.0 \
  ki:=0.0 \
  kd:=0.0 \
  use_rviz:=true
```

当前无扰动的一阶关节默认适合从纯 P 开始；I 用于补偿持续偏差但可能引入超调，D 主要用于带惯性的系统且需要注意噪声。关节角受 URDF 的正负 1.5708 rad 范围约束，速度命令也受到限幅。

Launch 文件结构、参数传递和进程边界见 [ROS 2 Launch 笔记](../../notes/concepts/ros2-launch.md)。

## 编码器与差速轮里程计

`WheelTicks` 自定义消息携带采样时间与左右轮累计编码器计数。`wheel_encoder_publisher` 生成确定性的理想计数，`odometry_node` 把计数差换算成左右轮路程，再用差速轮模型累计二维位姿：

```text
wheel_encoder_publisher → /wheel_ticks → odometry_node
                                           ├── /odom
                                           └── odom → base_link
```

运行时不要同时启动旧的 `position_tf_broadcaster`；它发布 `world → base_link`，会与里程计节点发布的 `odom → base_link` 形成冲突的 `base_link` 父坐标系。

终端一运行模拟编码器：

```bash
ros2 run point_robot_ros wheel_encoder_publisher
```

终端二运行里程计：

```bash
ros2 run point_robot_ros odometry_node
```

终端三检查 Topic 和 TF：

```bash
ros2 topic echo /odom --once
ros2 topic info /wheel_ticks
ros2 run tf2_ros tf2_echo odom base_link
```

默认左右轮每周期都增加 10 ticks，因此机器人沿 x 方向直行。把右轮增量改为 12 ticks 后，实际验证 `/odom` 的位置与航向同时变化，符合圆弧运动预期。12 个单元测试覆盖编码器换算、直行、旋转、圆弧、角度归一化、首帧初始化和连续累计；完整工作空间验证见前文“测试”。

当前仍是无噪声、无打滑的理想软件里程计，没有真实硬件、协方差或外部传感器校正。详细原理与证据见 [编码器与差速轮里程计笔记](../../notes/concepts/ros2-wheel-encoder-odometry.md)。

## 模拟相机与二维 LiDAR

`synthetic_camera_publisher` 发布 `320 × 240 mono8` 灰度图和理想 CameraInfo。图像包含灰度渐变和移动白条，两个 Topic 使用相同时间戳与 `camera_optical_frame`：

```text
/camera/image_raw   [sensor_msgs/msg/Image]
/camera/camera_info [sensor_msgs/msg/CameraInfo]
```

`camera_optical_frame` 通过固定关节连接到 `camera_link`，将机器人常规坐标转换为 x 向右、y 向下、z 向前的相机光学坐标。RViz Image 最初因请求 `RELIABLE` 而与 `BEST_EFFORT` 发布者不兼容；把 Reliability Policy 改为 Best Effort 后图像正常显示。

`synthetic_lidar_publisher` 在 `/scan` 发布 181 束、-90° 到 +90° 的 `sensor_msgs/msg/LaserScan`。`laser_frame` 固定安装在 `base_link` 的 `[0.05, 0, 0.24]`，RViz 能显示正前方 1.5 m、左前方 2.0 m 和其余方向 4.0 m 的扫描点。

Launch 参数按需启动传感器：

```bash
ros2 launch point_robot_ros point_robot.launch.py \
  velocity_x:=0.0 \
  use_camera:=true \
  use_lidar:=true \
  use_rviz:=true
```

9 个单元测试验证图像字节布局、移动条纹、非法尺寸、LiDAR 角度索引和非法扫描参数。相机与 LiDAR 还通过 Topic、QoS、TF 和 RViz 完成实际运行验证。当前数据是确定性的理想软件模式，不包含真实场景渲染、噪声、遮挡、运动畸变或硬件标定。详细原理见 [摄像头与二维激光雷达笔记](../../notes/concepts/ros2-camera-lidar-basics.md)。

## 状态估计基础

`ScalarKalmanFilter` 用一维位置模型验证状态估计的 Prediction 和 Correction。估计器保存位置估计、方差、最近一次 Innovation 和 Kalman Gain，并通过 `Q` 与 `R` 表达运动模型和位置观测的不确定性。

```bash
ros2 run point_robot_ros state_estimation_demo
```

演示中真实速度为 `1.0 m/s`，带偏差的运动模型使用 `0.9 m/s`，位置观测包含确定性正负噪声。实际结果为：

```text
Measurement RMSE: 0.5060
Prediction RMSE:  0.6205
Kalman RMSE:      0.2538
```

5 个单元测试覆盖预测、方差增长、数值更新、测量噪声对 Gain 的影响、重复观测和非法不确定性。当前实现不是 ROS 多传感器融合节点，也没有扩展到二维 EKF、IMU 或 SLAM；目标是建立能使用和排查现成融合系统所需的基础。详细说明见 [状态估计与一维 Kalman Filter](../../notes/concepts/state-estimation-kalman-basics.md)。

## 机器人安全基础

`safety_node` 位于控制器和执行器之间，只发布通过安全检查的线速度：

```text
/point_robot/cmd_vel_raw
            ↓
       safety_node ── /point_robot/set_emergency_stop
            ↓
/point_robot/cmd_vel_safe
            ↓
 safe_position_simulator
            ↓
 /point_robot/position → TF
```

启动节点：

```bash
ros2 run point_robot_ros safety_node \
  --ros-args \
  -p max_linear_speed:=1.0 \
  -p command_timeout:=0.5
```

单独启动 `safety_node` 只检查安全输出；若需要让它真正驱动模拟位置，先停止其他位置发布者，再启动完整链路：

```bash
ros2 launch point_robot_ros point_robot.launch.py use_safety:=true
```

安全模式会关闭自主匀速的 `position_publisher`，改由 `safe_position_simulator` 接收安全速度。初始无指令时不运动，`velocity_x` 只用于非安全模式。模拟器也有独立的指令超时检查。

在同一 ROS 环境的另一个终端持续发送低速命令，可观察位置增加；按 Ctrl+C 停止发送后观察位置保持不变：

```bash
ros2 topic pub --rate 10 /point_robot/cmd_vel_raw geometry_msgs/msg/Twist "{linear: {x: 0.2}}"
```

实际验证包括：速度限幅、指令断流停车、`NaN` 输入清空旧速度、急停停车，以及解除急停后仍需新指令才能运动。新增集成测试同时观察安全输出与连续位置消息，而非只检查安全话题。

13 个安全逻辑单元测试、原有安全 Topic 集成测试及新增安全运动集成测试均通过。当前执行端只是固定步长积分的教学模拟器，不包含真实制动过程；软件机制无法保证执行端进程或操作系统失效后的物理停车，不能替代硬件急停和驱动器使能回路。详细说明见 [ROS 2 机器人安全基础](../../notes/concepts/ros2-robot-safety-basics.md)。

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

`urdf/point_robot.urdf` 描述一个蓝色箱体底座、可绕 z 轴转动的摄像头，以及固定安装的二维激光雷达：

```text
base_link
├── camera_joint (revolute, -90° to 90°)
│   └── camera_link
│       └── camera_optical_frame
└── laser_frame
```

底座尺寸为 `0.6 × 0.4 × 0.2 m`，摄像头尺寸为 `0.12 × 0.08 × 0.08 m`。`camera_joint` 的安装点位于底座前方 `0.25 m`、上方 `0.28 m`，旋转范围约为 `-1.571～1.571 rad`。`camera_optical_frame` 提供图像算法使用的光学坐标约定；圆柱形 `laser_frame` 位于底座前方 `0.05 m`、上方 `0.24 m`。

URDF 由 `setup.py` 安装到功能包共享目录。Launch 从安装目录读取文件，并通过标准参数 `robot_description` 交给 `robot_state_publisher`。

```bash
check_urdf src/point_robot_ros/urdf/point_robot.urdf
ros2 launch point_robot_ros point_robot.launch.py
ros2 run tf2_ros tf2_echo base_link camera_link
ros2 run tf2_ros tf2_echo world camera_link
```

实际验证结果：

- `check_urdf` 成功解析，根 Link 为 `base_link`，分支包含 `camera_link → camera_optical_frame` 和 `laser_frame`。
- `/joint_states` 正确发布 `camera_joint` 的角度，`robot_state_publisher` 据此更新 `base_link → camera_link`。
- 关节角约为 `1.571 rad` 时，Yaw 为 90°，四元数约为 `[0, 0, 0.707, 0.707]`。
- 摄像头安装点平移保持 `[0.25, 0, 0.28]`，方向随关节角变化。
- TF2 能继续组合底座运动与摄像头转动，得到动态 `world → camera_link`。
- 最新完整测试由实际 WSL 终端确认通过，无 errors 或 failures。

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
- 运行时把 `velocity_x` 从 `0.0` 改为 `0.1` 后，两个 Link 沿 world 的 x 轴一起移动；摄像头安装点相对底座固定，但方向还可由 `camera_joint` 独立改变。
- 再把速度设为 `0.0` 后模型停止；调用 Reset Service 后模型返回原点附近。
- 保存配置后，`use_rviz:=true` 能自动恢复 Fixed Frame、RobotModel、TF 和观察视角。
- 关节 GUI 能实时驱动摄像头旋转，最新完整测试保持无 errors 或 failures。

RViz 的显示数据流和诊断边界见 [ROS 2 RViz 笔记](../../notes/concepts/ros2-rviz.md)。RViz 是可视化工具，不负责物理仿真、碰撞响应或机器人控制。

## 摄像头运动学

摄像头云台只有一个绕 z 轴旋转的关节变量 `yaw`。纯 Python 模块 `point_robot_ros/kinematics.py` 实现：

- 正运动学：由 `yaw` 和前向距离计算目标点在 `base_link` 中的位置。
- 简单逆运动学：由平面目标坐标通过 `atan2` 计算所需 `yaw`。
- 对非有限输入、负距离、与安装点重合的目标和超出关节范围的目标进行拒绝。

```text
x = 0.25 + distance × cos(yaw)
y = distance × sin(yaw)
z = 0.28
```

6 个单元测试覆盖 0°、90°、正逆往返和非法边界。实际交叉验证中，`yaw=-0.4 rad`、距离 `1 m` 时，Python 计算与 TF2 的 `base_link → camera_forward` 都得到约 `[1.171, -0.389, 0.280]`。

完整概念与证据见 [机器人运动学基础笔记](../../notes/concepts/robot-kinematics-basics.md)。当前 JointState 直接指定角度，尚未模拟电机和闭环控制。

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
