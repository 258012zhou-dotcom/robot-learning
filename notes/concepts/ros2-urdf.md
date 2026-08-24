# ROS 2 URDF 机器人模型

## 快速复习

- URDF 使用 XML 描述机器人的刚体、关节、外观、碰撞和惯性属性。
- `link` 是刚体，`joint` 连接父 Link 和子 Link，并定义相对位姿与运动类型。
- `visual` 用于显示，`collision` 用于碰撞，`inertial` 用于动力学计算。
- `robot_state_publisher` 读取 `robot_description`，把 URDF 关节关系发布到 TF2。
- 固定关节发布到 `/tf_static`；可动关节还需要 `/joint_states` 才能计算当前 TF。
- `check_urdf` 只验证语法和树结构，不验证 RViz 外观或物理参数是否合理。

## Link 与局部坐标系

每个 Link 都有自己的坐标系。当前模型采用机器人常见约定：x 向前、y 向左、z 向上。

`base_link` 原点位于底座地面中心，箱体尺寸为 `0.6 × 0.4 × 0.2 m`。visual、collision 和质心都沿 z 上移 `0.1 m`，因此箱体底面位于 `z=0`。

`camera_link` 的几何中心与自身坐标原点重合。它在机器人上的安装位置由 `camera_joint` 决定，而不是由 camera Link 自己决定。

## Visual、Collision 与 Inertial

这三部分可以使用不同的形状和原点：

- visual 可以使用精细网格和颜色，方便人观察。
- collision 通常使用简化几何体，降低碰撞计算成本。
- inertial 包含质心、质量和对转动的惯性。

当前两个 Link 都使用箱体。对齐坐标轴的对称箱体交叉惯量 `ixy`、`ixz`、`iyz` 为零；对角项由质量和边长决定。RViz 不使用惯量，但后续物理仿真会使用，因此模型中提前保留了合理近似值。

## Joint 与坐标关系

当前固定关节为：

```xml
<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.25 0 0.28" rpy="0 0 0"/>
</joint>
```

它表示 `camera_link` 位于 `base_link` 前方 `0.25 m`、上方 `0.28 m`，方向一致。`fixed` 表示相对位姿不随时间变化，所以不需要 `joint_state_publisher`。

若关节类型改为 `revolute`、`continuous` 或 `prismatic`，还要声明转轴、运动范围，并提供实时 JointState。URDF 描述结构和约束，不负责自己计算控制命令。

## 从文件到 TF

本项目的流程是：

```text
point_robot.urdf
  → setup.py 安装到 share/point_robot_ros/urdf
  → Launch 读取 XML
  → robot_description 参数
  → robot_state_publisher
  → base_link → camera_link
```

同时，`position_tf_broadcaster` 发布动态的 `world → base_link`。TF2 将两段关系组合成持续更新的 `world → camera_link`。

## 验证证据与边界

`check_urdf` 已确认根 Link 为 `base_link`，并正确连接一个子 Link `camera_link`。`tf2_echo` 验证固定平移为 `[0.25, 0, 0.28]`，完整 `world → camera_link` 也能随底座运动更新。自动测试保持 8 tests、0 errors、0 failures、1 skipped。

当前模型仍是简单几何体，没有轮子、可动关节、网格资源或真实传感器参数。颜色、尺寸比例、遮挡关系和运动显示需要在 RViz 中继续验证。
