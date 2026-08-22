# ROS 2 Parameter

## 快速复习

- Parameter 是节点的具名配置，不是持续传输数据的 Topic。
- `declare_parameter()` 声明名称、类型和默认值，`get_parameter()` 读取当前值。
- `--ros-args -p name:=value` 在节点启动时覆盖默认值。
- `ros2 param get/set` 可以在运行时查询或请求修改参数。
- 参数服务器中的值改变，不等于程序缓存的成员变量自动改变；动态参数需要回调同步程序状态。
- 回调应先验证整组修改，再统一应用，并用 `SetParametersResult` 接受或拒绝请求。

## 参数的基本流程

节点先声明参数：

```python
self.declare_parameter("velocity_x", 0.5)
```

然后读取它：

```python
self._velocity_x = float(
    self.get_parameter("velocity_x").value
)
```

启动时可以覆盖默认值：

```bash
ros2 run point_robot_ros position_publisher \
  --ros-args \
  -p velocity_x:=1.0
```

常用查询命令：

```bash
ros2 param list /position_publisher
ros2 param get /position_publisher velocity_x
ros2 param set /position_publisher velocity_x 2.0
```

`use_sim_time` 是 ROS 2 常见的系统参数，用于选择系统时间或仿真时钟。

## 启动参数与动态参数

启动参数只需在构造节点时读取一次。动态参数则必须在节点运行期间处理更新请求：

```python
self.add_on_set_parameters_callback(self._update_parameters)
```

当前项目把参数分成两类：

- `initial_x`、`timer_period`：启动专用，运行时修改会被明确拒绝。
- `velocity_x`：支持动态修改，回调同步更新 `self._velocity_x`。

`timer_period` 不能只修改参数值，因为已经创建的 ROS Timer 不会因此自动改变周期；若要支持动态周期，需要显式取消或重建 Timer。

## 回调的可靠性原则

参数回调可能一次收到多个参数。可靠做法是：

1. 先验证所有名称、类型和取值范围。
2. 任何一个参数非法时，返回 `successful=False`，并给出原因。
3. 全部合法后再修改程序内部状态。

先验证、后应用可以避免一组参数最终被拒绝，但前半部分已经改变程序状态。

## 当前项目的实际验证

发布周期为 `0.2 s`、初始速度为 `1.0` 时，位置步长为：

```text
1.0 × 0.2 = 0.20
```

运行时把 `velocity_x` 改为 `2.0` 后，日志出现 `Updated velocity_x to 2.00`，后续位置为：

```text
4.40 → 4.80 → 5.20 → 5.60
```

步长变为 `0.40`，证明参数回调实际改变了节点行为。运行时修改 `timer_period` 返回 `timer_period is startup-only`，证明非法更新被明确拒绝。

现有自动测试主要检查代码和包规范；动态行为是通过真实运行节点和 CLI 修改参数验证的。
