# ROS 2 摄像头与二维激光雷达基础

## 快速复习

- `sensor_msgs/msg/Image` 保存像素数据，`CameraInfo` 保存相机内参与畸变参数；二者需要相同时间戳和光学坐标系。
- 针孔模型用 `u = fx X/Z + cx`、`v = fy Y/Z + cy` 把三维点投影到像素平面。
- ROS 相机光学坐标系通常是 x 向右、y 向下、z 向前，与机器人机体坐标约定不同。
- `sensor_msgs/msg/LaserScan` 按固定角度顺序保存一圈或一段扫描距离，第 `i` 束角度是 `angle_min + i × angle_increment`。
- 相机和 LiDAR 常用 `BEST_EFFORT` 传感器 QoS；订阅者若强制请求 `RELIABLE`，DDS 会拒绝不兼容连接。
- 图像、扫描和模型能否在 RViz 正确对齐，取决于时间戳、`frame_id`、TF 和传感器内参，而不只是 Topic 是否存在。

## 图像消息与内存布局

`sensor_msgs/msg/Image` 的关键字段包括：

- `header.stamp`：采集时间。
- `header.frame_id`：相机光学坐标系。
- `height`、`width`：图像尺寸。
- `encoding`：像素通道和每通道数据类型。
- `step`：一行图像占用的字节数。
- `data`：按行连续排列的原始像素字节。

常见编码：

- `mono8`：单通道 8 位灰度，每像素 1 字节。
- `rgb8`、`bgr8`：三通道彩色，每像素 3 字节。
- `16UC1`：单通道 16 位无符号整数，常见于以毫米表示的深度图。
- `32FC1`：单通道浮点数，常见于以米表示的深度图。

当前模拟图像为 `320 × 240 mono8`，因此：

```text
step = 320 bytes
data length = 320 × 240 = 76800 bytes
```

模拟节点生成从左到右的灰度渐变，并让 8 像素宽的白色条纹每帧向右移动 5 像素。它验证的是 Image 消息结构和传输链路，不模拟真实光照、镜头或场景渲染。

## CameraInfo 与针孔模型

图像只给出像素值，`sensor_msgs/msg/CameraInfo` 还描述相机如何把三维射线投影为像素。第一轮重点是内参矩阵：

```text
K = [fx  0 cx
      0 fy cy
      0  0  1]
```

三维点 `(X, Y, Z)` 投影到像素 `(u, v)`：

```text
u = fx X/Z + cx
v = fy Y/Z + cy
```

`fx`、`fy` 是像素单位焦距，`cx`、`cy` 是主点。`D` 和 `distortion_model` 描述镜头畸变，`R` 用于校正，`P` 是投影矩阵。当前模拟相机使用零畸变和理想内参，不代表真实相机已经标定。

Image 与 CameraInfo 使用相同的时间戳和 `camera_optical_frame`，表示它们属于同一帧采样。真实多传感器系统还必须考虑时间同步、曝光延迟和标定误差。

## 相机坐标系

机器人机体坐标通常是 x 向前、y 向左、z 向上；相机光学坐标是：

```text
x 向右
y 向下
z 向前
```

项目通过固定关节建立：

```text
base_link
└── camera_link
    └── camera_optical_frame
```

`camera_link → camera_optical_frame` 使用 `rpy = [-90°, 0°, -90°]`，完成轴映射：optical x 对应 `camera_link` 的 -y，optical y 对应 -z，optical z 对应 x。这个变换只改变坐标表达，不改变相机物理位置。

## LaserScan 的角度与距离

二维激光雷达把一帧扫描保存为按角度排列的 `ranges`：

```text
θ_i = angle_min + i × angle_increment
x_i = range_i × cos(θ_i)
y_i = range_i × sin(θ_i)
```

关键字段：

- `angle_min`、`angle_max`：扫描范围。
- `angle_increment`：相邻激光束角度差。
- `range_min`、`range_max`：有效测距范围。
- `scan_time`：完成一帧扫描的时间。
- `time_increment`：相邻激光束之间的采样时间差。
- `ranges`：每束激光的距离。
- `intensities`：可选反射强度。

当前模拟扫描从 -90° 到 +90°，共 181 束，角分辨率约为 1°。正前方 ±10° 距离为 1.5 m，左前方 35°～50° 距离为 2.0 m，其余方向为 4.0 m。`laser_frame` 通过固定关节安装在 `base_link` 前方 0.05 m、上方 0.24 m。

## QoS 与 RViz

相机和 LiDAR 都使用 `qos_profile_sensor_data`，其可靠性为 `BEST_EFFORT`。高频传感器数据通常更重视及时收到最新样本，而不是等待旧样本重传。

实际验证中，RViz Image 最初请求 `RELIABLE`，发布者日志报告 `incompatible QoS`，因此没有图像。把 RViz 的 Reliability Policy 改为 `Best Effort` 后连接恢复。这说明 Topic 名称和消息类型相同仍不够，QoS 也必须兼容。

RViz 的 `Image` 显示只展示二维像素；`Camera` 显示还结合 CameraInfo 与 TF。`LaserScan` 显示根据每束角度、距离和 `laser_frame` 计算扫描点。RViz 只可视化已有消息，不生成真实场景、碰撞或传感器测量。

## 当前项目数据流与证据

```text
synthetic_camera_publisher
├── /camera/image_raw
└── /camera/camera_info

synthetic_lidar_publisher
└── /scan

robot_state_publisher
└── world → base_link → camera_link → camera_optical_frame
                  └── laser_frame
```

实际完成的验证包括：

- RViz Image 显示灰度渐变和移动白条。
- CameraInfo 发布理想内参，图像与内参使用相同时间戳和 frame。
- `camera_optical_frame` 在 TF 中存在，轴方向符合光学约定。
- RViz LaserScan 显示前方、左前方和背景三段距离。
- RobotModel、相机坐标系、LiDAR 坐标系和扫描点能随底座一起运动。
- 9 个单元测试验证图像字节数、行布局、条纹移动、非法尺寸、LiDAR 角度索引和非法扫描参数。
- 完整工作空间测试为 50 tests、0 errors、0 failures、1 skipped。

当前结果仍是确定性软件数据：图像不是场景渲染，扫描障碍物会随机器人一起移动，没有噪声、遮挡、运动畸变、真实标定或硬件时间戳。已完成的一维状态估计另外使用带噪声位置观测；见[Kalman Filter 笔记](state-estimation-kalman-basics.md)，不要把它当作真实摄像头或雷达融合已经完成。
