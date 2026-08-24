# ROS 2 RViz 可视化

## 快速复习

- RViz 是 ROS 2 数据可视化和诊断工具，不是物理仿真器。
- Fixed Frame 是所有显示数据最终转换到的参考坐标系。
- RobotModel 从 `/robot_description` 读取 URDF 外观，再通过 TF 确定每个 Link 的当前位姿。
- TF 显示项用于检查坐标轴、父子树和坐标系是否连通。
- `.rviz` 配置保存显示项、Fixed Frame、视角和面板布局，方便复现实验界面。

## RViz 如何显示机器人

当前模型显示依赖两类数据：

```text
/robot_description → RobotModel 获取几何、颜色和 Link 结构
/tf 与 /tf_static → RobotModel 获取每个 Link 的当前位姿
```

只有 URDF 而没有 TF 时，RViz 知道机器人长什么样，却不知道各 Link 应显示在哪里。只有 TF 而没有 URDF 时，可以看到坐标轴，却没有机器人外观。

当前 Fixed Frame 为 `world`。RViz 将 `base_link` 和 `camera_link` 都转换到 world 后绘制，因此能看到整个机器人沿世界 x 轴移动。

## 主要显示项

- Grid：提供世界平面的尺度和方向参考。
- RobotModel：显示 URDF 的 visual 几何体；当前读取 `/robot_description`。
- TF：显示 `world → base_link → camera_link` 的坐标树、名称、箭头和坐标轴。

常见红色状态不一定表示程序崩溃。Fixed Frame 不存在通常是名称配置错误；某个 Link 缺少 Transform 通常是 TF 尚未发布、树不连通或时间查询失败；RobotModel 无内容则应检查 `/robot_description` 和模型资源路径。

## 当前项目的动态验证

系统以 `velocity_x=0.0` 启动时，机器人停在 world 原点。运行时把速度改为 `0.1` 后，位置 Topic 驱动动态 `world → base_link`，底座和摄像头在 RViz 中一起移动。摄像头安装点跟随底座，`camera_joint` 角度则可由 JointState 独立改变。速度恢复为零后模型停止，Reset Service 会使模型返回原点附近。

这条可视化链路覆盖了 Parameter、Topic、Service、TF2、URDF、robot_state_publisher 与 RViz，但画面正确不等于控制稳定或物理模型真实。

## 配置保存与 Launch

配置文件保存为 `rviz/point_robot.rviz`，由 `setup.py` 安装到功能包共享目录。Launch 的 `use_rviz` 参数通过 `IfCondition` 控制 RViz 节点：默认 `false` 适合后台和测试，显式设置为 `true` 时通过 `-d` 加载保存配置。

## 当前边界

RViz 不计算重力、摩擦、碰撞、传感器噪声或执行器响应，也不会主动改变机器人状态。当前显示的是软件生成的位置和简化 URDF；后续若要验证动力学与接触，需要 Gazebo、MuJoCo 或其他仿真器，真实机器人结果则还需要硬件运行证据。
