# ROS 2 在 WSL 与 TUN 环境中的 DDS 发现问题

## 问题环境

- WSL Ubuntu 22.04
- ROS 2 Humble
- 默认中间件：Fast DDS（`rmw_fastrtps_cpp`）
- Windows 代理软件启用 TUN，WSL 中同时存在多个网络接口
- 多播测试的数据来自 `198.18.0.1`

## 现象

- ROS 2 发布节点能够正常启动并持续发布。
- `ros2 node list` 长时间无法返回。
- 同一 WSL 中的订阅节点收不到发布节点消息。
- 设置 `ROS_LOCALHOST_ONLY=1` 后问题仍然存在。

## 诊断证据

1. `ros2 multicast send/receive` 成功，接收端收到来自 `198.18.0.1` 的 `Hello World!`。这排除了 UDP 多播被完全阻断。
2. ROS 官方 `demo_nodes_cpp talker` 可以发布，但使用默认 Fast DDS 的 `listener` 收不到消息。这排除了项目节点代码是唯一原因。
3. 安装并选择 Cyclone DDS 后，官方 talker/listener 成功通信。
4. 使用相同 Cyclone DDS 配置后，项目的位置发布节点与订阅节点也成功通信。

因此，问题定位为 Fast DDS 在当前 WSL 多网卡/TUN 配置下的发现或数据通道异常。该结论只针对当前环境，不表示 Fast DDS 本身普遍不可用。

## 当前解决方法

安装 Cyclone DDS：

```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

每个参与通信的终端都使用相同配置：

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=31
```

- `RMW_IMPLEMENTATION` 选择 ROS 2 使用的 DDS 实现。
- `ROS_DOMAIN_ID` 隔离不同 ROS 系统；需要通信的进程必须使用相同值。
- 环境变量只影响当前终端及其子进程，新终端需要重新设置。
- 构建本地工作空间后，还要继续执行 `source install/setup.bash`。

## 排查顺序

遇到 ROS 2 节点互相发现失败时，依次检查：

1. 两个终端的 `ROS_DOMAIN_ID` 与 `RMW_IMPLEMENTATION` 是否一致。
2. `ros2 multicast send/receive` 是否成功。
3. 官方 demo talker/listener 是否通信。
4. 再检查项目节点和消息配置。
5. 最后才考虑更换 DDS、绑定网络接口或修改防火墙。

不要仅凭发布节点打印日志就判断 Topic 已成功；必须在订阅端看到实际消息。

## CLI daemon 与 DDS 正常通信的区别

### 已验证现象

项目位置发布者运行时：

- `ros2 topic hz /point_robot/position` 能收到约 `10 Hz` 的消息。
- `ros2 node list` 和 `ros2 topic list -t` 却长时间不返回。
- `ps` 查不到正在运行的 `_ros2_daemon` 进程。
- 加入 `--no-daemon --spin-time 3` 后，可以立即发现发布者节点和位置 Topic。

这些证据说明发布者、Topic 数据和 Cyclone DDS 发现本身正常，异常只发生在 ROS 2 CLI 使用的后台 daemon。`ros2 topic hz` 会直接创建订阅者接收数据，而 `node list`、`topic list` 默认可能通过 daemon 查询缓存，因此两类命令的结果可以不同。

### 恢复方法

先确认当前终端配置：

```bash
echo "$RMW_IMPLEMENTATION"
echo "$ROS_DOMAIN_ID"
```

本项目应分别得到 `rmw_cyclonedds_cpp` 和 `31`。然后在正确环境中重新启动 daemon：

```bash
timeout 10s ros2 daemon start
```

本次启动成功后，普通的 `ros2 node list` 和 `ros2 topic list -t` 恢复正常。`timeout` 只负责避免命令无限等待，不是修复手段；真正起作用的是新 daemon 继承了当前终端正确的 RMW 和 Domain ID。

若 daemon 再次异常，可以临时绕过它：

```bash
ros2 node list --no-daemon --spin-time 3
ros2 topic list -t --no-daemon --spin-time 3
```

切换 `RMW_IMPLEMENTATION` 或 `ROS_DOMAIN_ID` 后，应停止旧 daemon，并在新环境中重新启动。不要把 daemon 查询失败误判为 Topic 数据一定中断。
