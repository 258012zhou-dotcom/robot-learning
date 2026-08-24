# 机器人运动学基础

## 快速复习

- 关节空间用关节变量描述机器人，例如摄像头云台角 `q = yaw`。
- 任务空间描述末端或传感器的位置和方向。
- 正运动学由关节变量计算任务空间位姿；逆运动学由目标位姿反求关节变量。
- URDF 定义关节轴、安装位姿和限制，JointState 给出当前角度，`robot_state_publisher` 计算 Link TF。
- 运动学不考虑质量、力矩、摩擦、加速度和电机响应，这些属于动力学与控制问题。

## 当前单自由度模型

`camera_joint` 位于 `base_link` 的 `[0.25, 0, 0.28]`，绕 z 轴旋转，范围约为 ±90°。它只有一个自由度，所以关节空间变量是一个标量：

```text
q = yaw
```

摄像头安装点不随 q 改变，变化的是摄像头坐标轴方向。摄像头前方距离 `d` 的点在 `base_link` 中为：

```text
x = 0.25 + d cos(q)
y =        d sin(q)
z = 0.28
```

这就是当前系统的正运动学。`q=0` 时朝向正 x，`q=π/2` 时朝向正 y。

## 简单逆运动学

目标点在 `base_link` 中为 `(target_x, target_y)` 时，先减去摄像头安装点：

```text
dx = target_x - 0.25
dy = target_y
q = atan2(dy, dx)
```

若目标与安装点重合，方向没有定义。若结果超出 ±90°，当前关节无法到达该朝向。逆运动学不只要求公式有解，还要检查关节限制和目标是否可达。

## ROS 2 中的执行链

```text
joint_state_publisher 或 GUI
  → /joint_states
  → robot_state_publisher
  → URDF 正运动学
  → base_link → camera_link TF
  → RViz
```

后台模式使用无界面的 JointState 发布器给出默认角度；交互模式使用 GUI 滑块。Launch 通过互斥条件保证两者不会同时发布冲突状态。

## 测试与交叉验证

`kinematics.py` 是不依赖 ROS 的纯数学模块。6 个单元测试验证 0°、90°、正逆往返、负距离和不可达目标。单元测试失败表示公式或边界逻辑错误，不代表 DDS 故障。

实际验证中，`q=-0.4 rad`、`d=1 m` 时，Python 得到约 `[1.171, -0.389, 0.280]`。TF2 通过 `base_link → camera_link → camera_forward` 组合得到相同结果，证明数学实现与 ROS 2 坐标系统一致。

## 当前边界

JointState GUI 直接指定关节状态，并不模拟控制器驱动电机到达目标。当前逆运动学只解决平面内单关节朝向，不包含多关节、多解、奇异性、雅可比矩阵或数值 IK。这些会在更复杂机械臂模型中继续学习。
