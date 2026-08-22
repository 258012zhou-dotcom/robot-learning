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

### 深入定位结果

进一步检查确认，这次 CLI 卡住与前面的 Fast DDS 通信失败不是同一个问题：

- Windows 用户配置启用了 WSL `networkingMode=mirrored`。
- ROS 2 Domain 31 的 CLI daemon 使用本机 TCP 端口 `127.0.0.1:11542`，其中 `11542 = 11511 + 31`。
- daemon 不存在时，`strace` 显示 CLI 阻塞在连接 `127.0.0.1:11542`。
- `ip route get 127.0.0.1` 显示连接先进入 WSL 镜像网络的 `loopback0` 和路由表 127，而不是立即在 Linux 本机返回“端口未监听”。
- 多个未监听的本机端口都会超时；临时启动本机监听服务后，同一地址又能正常连接。
- daemon 一旦提前启动，普通的 `node list`、`param list` 和 `param get` 都能正常工作。

因此，直接原因是：ROS 2 CLI 假设“daemon 不存在时，连接本机端口会立即失败”，但当前 WSL 镜像网络会让这个未监听连接等待超时。`--no-daemon` 能绕过问题，但不适合作为长期日常用法。

### 当前长期方案

项目提供按需激活脚本：

```bash
cd ~/AI_Project/robot-learning
source scripts/activate_ros2_project.sh
```

它只配置当前终端，统一加载 ROS 2、工作空间、Cyclone DDS 和 Domain 31，不污染普通 Conda 环境。

用户级 systemd 服务持续维护 Domain 31 的 daemon：

```bash
systemctl --user status ros2-cli-daemon-domain31.service
systemctl --user restart ros2-cli-daemon-domain31.service
journalctl --user -u ros2-cli-daemon-domain31.service -n 50
```

服务定义位于：

```text
projects/ros2_point_robot_ws/systemd/ros2-cli-daemon-domain31.service
```

在新的本地副本中首次安装服务：

```bash
cd ~/AI_Project/robot-learning
systemctl --user link "$PWD/projects/ros2_point_robot_ws/systemd/ros2-cli-daemon-domain31.service"
systemctl --user enable --now ros2-cli-daemon-domain31.service
```

它使用 Cyclone DDS、Domain 31 和 7 天无活动超时；若进程退出，systemd 会自动重启。最终验证中，systemd 主进程与 `127.0.0.1:11542` 的监听进程一致，普通 ROS 2 查询命令均成功。

如需完整撤销：

```bash
systemctl --user disable --now ros2-cli-daemon-domain31.service
rm ~/.config/systemd/user/ros2-cli-daemon-domain31.service
systemctl --user daemon-reload
```

### 临时恢复方法

先确认当前终端配置：

```bash
echo "$RMW_IMPLEMENTATION"
echo "$ROS_DOMAIN_ID"
```

本项目应分别得到 `rmw_cyclonedds_cpp` 和 `31`。然后在正确环境中重新启动 daemon：

```bash
timeout 10s ros2 daemon start
```

daemon 启动成功后，普通的 `ros2 node list` 和 `ros2 topic list -t` 应恢复正常。`timeout` 只负责避免命令无限等待，不是修复手段；真正起作用的是 daemon 已经监听本机端口并继承了正确的 RMW 和 Domain ID。

若 daemon 再次异常，可以临时绕过它：

```bash
ros2 node list --no-daemon --spin-time 3
ros2 topic list -t --no-daemon --spin-time 3
```

切换 `RMW_IMPLEMENTATION` 或 `ROS_DOMAIN_ID` 后，应停止旧 daemon，并在新环境中重新启动。不要把 daemon 查询失败误判为 Topic 数据一定中断。
