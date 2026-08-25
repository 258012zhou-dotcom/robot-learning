# ROS 2 最小集成测试

## 快速复习

- 单元测试隔离验证函数或类；集成测试验证多个真实组件能否协同工作。
- `launch_testing` 可以启动真实 ROS 2 节点，再让测试代码通过 DDS 与节点通信。
- `ReadyToTest()` 表示被测进程已经交给 Launch 管理，可以开始执行测试断言。
- 测试使用独立 `ROS_DOMAIN_ID`，避免日常节点或残留进程污染结果。
- lint 通过只说明代码规范正常，不能证明 Topic、Service 或 Action 能通信。

## 当前测试的执行流程

`test_position_topic_launch.py` 的流程是：

1. Launch 启动 `position_publisher`，并传入测试专用参数。
2. 测试进程初始化 `rclpy`，创建临时订阅节点。
3. 临时节点订阅 `/point_robot/position`。
4. `rclpy.spin_once()` 驱动回调，最多等待 5 秒。
5. 收到至少三条消息后检查内容和运动规律。
6. 测试销毁订阅和临时节点，Launch 负责停止被测进程。

这条路径经过了节点启动、参数传递、ROS 消息类型、Topic 名称、DDS 发现与传输、订阅回调和发布者运动逻辑，因此比直接调用 Python 方法覆盖的范围更大。

`test_safety_node_launch.py` 使用相同结构启动 `safety_node`。临时测试节点向 `/point_robot/cmd_vel_raw` 发布超限的 `Twist`，订阅 `/point_robot/cmd_vel_safe`，先确认 `2.0` 被限制为 `1.0`，停止发布后再确认 Watchdog 使输出归零。它验证的是安全逻辑经过 ROS Topic 和定时器后的真实协作，而不只是直接调用 Python 类。

## 关键断言

- `len(received_messages) >= 3`：节点确实持续发布，而不是只偶然收到一条消息。
- 所有 `y == 0.0`：当前点机器人仍沿 x 轴运动。
- 最后一条 `x` 大于第一条：运动方向正确。
- 相邻 `x` 差约为 `0.05`：验证 `velocity_x=1.0` 与 `timer_period=0.05` 被正确应用。

若没有收到消息，可能是节点启动失败、Topic 或类型不一致、DDS 配置异常。若收到消息但步长错误，则更可能是参数传递或运动更新逻辑错误。

## 隔离与当前边界

本项目日常使用 Domain 31，测试临时使用 Domain 132。不同 Domain 的节点互不可见，因此正在运行的普通发布者不会混入测试数据。

当前自动集成测试覆盖位置 Topic，以及安全速度 Topic 的限幅和超时停车。Service 重置、软件急停、Action 成功/取消/并发拒绝，以及完整 Launch，已有手动运行证据，但尚未全部转化为自动集成测试。当前阶段保留这个边界，避免过早建立复杂测试框架。
