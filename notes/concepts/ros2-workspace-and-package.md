# ROS 2 工作空间与功能包

## 快速复习

- 工作空间（workspace）组织一个或多个功能包及其构建结果。
- 功能包（package）是 ROS 2 代码、依赖、资源和运行入口的基本单位。
- `package.xml` 描述 ROS 元数据和依赖；`setup.py` 描述 Python 安装规则。
- `colcon build` 生成构建结果；`source install/setup.bash` 将结果叠加到当前终端。
- 构建成功、规范测试通过和功能运行正确是三个不同层次的验证。
- Topic 是发布者与订阅者之间的异步数据通道；双方需要匹配 Topic 名称、消息类型和兼容的 QoS。

## 工作空间结构

```text
workspace/
├── src/       # 源码
├── build/     # 中间文件
├── install/   # 安装结果与环境脚本
└── log/       # 构建和测试日志
```

项目只提交 `src/` 和说明文档。其余目录由 colcon 生成，可以重新构建。

## Python 功能包的关键文件

- `package.xml`：包名、版本、维护者、许可证、运行依赖、测试依赖和构建类型。
- `setup.py`：安装 Python 模块、ament 索引资源、`package.xml` 和节点命令入口。
- `setup.cfg`：把节点脚本安装到 `lib/<package_name>`，供 `ros2 run` 查找。
- `resource/<package_name>`：将包登记到 ament 资源索引。
- `<package_name>/__init__.py`：Python 模块入口。

外层 `point_robot_ros/` 是 ROS 功能包根目录，内层同名目录是可以被 Python 导入的模块。

## 环境叠加

```text
/opt/ros/humble/setup.bash
  → workspace/install/setup.bash
  → 当前终端能发现系统包和本地包
```

环境变量属于当前进程。构建后不执行第二次 `source`，ROS 就可能找不到新包；删除工作空间也不会自动清除旧终端中的路径。

本项目根目录的机器学习代码使用 Conda，ROS 2 Humble 使用 `/usr/bin/python3`。两个环境使用独立终端，避免 Python 包和 pytest 插件冲突。

## 构建与测试

`colcon build --symlink-install` 扫描 `src/` 并构建包。符号链接安装便于开发 Python 源码，但修改包元数据、资源或命令入口后仍应重新构建。

`colcon test` 运行包注册的测试，`colcon test-result --verbose` 汇总结果。当前三个模板测试分别检查：

- Flake8：Python 代码风格。
- PEP 257：文档字符串格式。
- Copyright：版权头，当前主动跳过。

测试中的 `assert rc == 0` 要求检查工具返回成功状态。规范测试通过不能证明 ROS 节点已经通信，后续仍需单元测试和实际运行验证。

## 当前项目证据

- 用户亲自创建了工作空间和 `ament_python` 功能包。
- `point_robot_ros` 构建成功，并能由 `ros2 pkg prefix` 发现。
- 包使用独立 `pytest.ini`，避免读取根项目的 pytest 配置。
- 实际测试结果为 3 tests、0 errors、0 failures、1 skipped。
- `position_publisher` 以约 `10 Hz` 发布 `geometry_msgs/msg/Point`，位置的 `x` 每次增加 `0.05`。
- `position_subscriber` 通过回调函数接收 `/point_robot/position` 的新消息。
- 实际运行时 ROS Graph 显示 1 个发布者和 1 个订阅者，订阅端连续收到位置数据。

## Node 与 Topic 通信流程

```text
定时器
  → PositionPublisher 创建 Point 消息
  → /point_robot/position
  → Cyclone DDS 传输
  → PositionSubscriber 的回调函数
  → 输出收到的位置
```

发布者调用 `create_publisher`，订阅者调用 `create_subscription`。订阅者不会主动轮询 Topic；ROS 2 executor 收到消息后调用注册的回调函数。

双方当前使用的队列深度都是 `10`。它表示接收处理不及时期间可以保留的消息数量，不表示新订阅者能够读取连接前的十条历史消息。默认 volatile durability 下，订阅者通常只收到加入之后发布的新消息。

Topic 适合连续、异步的数据流，例如传感器读数、机器人状态和速度指令。它不要求发布者等待订阅者处理完成，也不会像函数调用一样直接返回结果。

## Service 请求与响应

Service 用于一次请求对应一次响应的操作：

```text
客户端请求
  → /point_robot/reset
  → PositionPublisher 的服务回调
  → 将内部位置重置为 0
  → 返回 success 和 message
```

本项目使用 `std_srvs/srv/Trigger`。它的请求部分为空，响应包含 `bool success` 和 `string message`，适合“重置”“触发保存”等不需要请求参数的命令。

服务端通过 `create_service` 注册接口和回调。客户端调用时，ROS 2 executor 执行服务回调；回调修改节点状态并返回响应。当前单线程 executor 会依次处理定时器与服务回调，因此本实验中二者不会同时修改位置。

实际验证中，发布位置先增长到 `25.10`，服务返回 `success=True` 后，下一条 Topic 消息变为 `0.00`，随后继续以 `0.05` 递增。这同时证明了请求得到响应、内部状态被修改以及 Topic 继续运行。

Python 客户端的调用流程为：

```text
wait_for_service
  → call_async 返回 Future
  → spin_until_future_complete 处理 ROS 事件
  → future.result 取得响应
```

`Future` 表示尚未完成、稍后才会得到的结果。`call_async` 发出请求后立即返回，客户端必须继续处理 ROS 事件，响应到达后 Future 才会完成。项目中的 `reset_client` 已实际返回成功响应并正常退出。

## Action 目标、反馈与结果

Action 适合需要一段时间完成、需要过程反馈的任务：

```text
MoveToPosition Goal
  → Action Server 接受目标
  → 循环更新位置
  → 发布位置 Topic
  → 发布 current_x 与 remaining_distance Feedback
  → 到达目标
  → 返回 success、final_x 与 message Result
```

项目把通信协议和节点实现分成两个包：

- `point_robot_interfaces` 使用 `ament_cmake` 和 rosidl 定义、生成 `MoveToPosition.action` 类型。
- `point_robot_ros` 使用 `ament_python` 实现 Action Server、Action Client 和位置订阅者。

接口包只规定 Goal、Result 和 Feedback 的字段，不实现运动算法。构建后会生成 Python、C、C++ 和 DDS 类型支持，因此不同语言或不同机器上的节点可以共享同一通信协议。

当前运动更新规则为：

```text
每步最大位移 = max_speed × 0.1
实际步长 = min(每步最大位移, 剩余距离)
```

Action Client 需要处理两个 Future：

```text
send_goal_async
  → goal Future：等待目标被接受或拒绝
  → get_result_async
  → result Future：等待任务最终完成
```

Feedback 通过独立回调在等待 Result 期间持续到达。实际验证中，CLI Client 和 Python Client 都能完成目标；位置订阅者观察到相同运动过程，最终 Result 为 `SUCCEEDED`。

当前版本的限制：

- 使用单线程、阻塞式执行循环。
- 明确拒绝取消请求。
- 尚未定义同时收到多个 Goal 时的调度策略。

因此当前只记录 Goal、Feedback、Result 与 Topic 联动已验证，取消和并发仍是知识空白。

## Topic、Service 与 Action 的选择

- 连续状态、传感器数据和控制流使用 Topic。
- 有明确完成结果的一次性操作使用 Service。
- 耗时较长、需要进度反馈或取消能力的任务应使用 Action。

## 易错点

- 在 Conda 环境中运行 ROS 2 Python 包。
- 构建后忘记加载 `install/setup.bash`。
- 把生成的 `build/`、`install/` 和 `log/` 提交 Git。
- 修改 `setup.py` 中的节点入口后忘记重新构建。
- 看到发布日志或规范测试通过，就误认为进程间通信已经成功。
- 把 QoS 队列深度误解成跨时间保存的历史消息数量。
- 只检查 Service 的成功响应，却不验证它是否真的改变了系统状态。
- 把 Action Goal 被接受误认为任务已经成功完成。
- 实现阻塞执行循环，却同时宣称支持可靠取消。
