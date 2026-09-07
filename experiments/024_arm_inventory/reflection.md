# 实验 024 反思

## 当前结论

实验 024 已在真实 AGILEX PIPER 上完成第一次只读反馈；通信链路可用，但使能状态和夹爪单位仍需核对，因此暂不进入运动控制。

## 从资料和电脑中确认的事实

- 铭牌已确认机械臂为 AGILEX PIPER；完整序列号不写入公开仓库。
- `can0` 是机械臂 CAN 接口。
- `piper_sdk 0.6.1` 已安装到个人 `xvla` 环境；该环境作为后续机械臂实验环境。
- 历史文档记录了 Piper ROS 2 节点和 RealSense 启动方式，但当前 SSH 环境没有发现已经加载的 ROS 2 Piper 包。
- ACT 的连接、校准、动作发送和退出清理不是纯只读流程，不能用于第一次检查。
- ACT 适配代码中的单位换算存在未完成痕迹，控制前必须核对。

## 首次真机只读结果

- 使用 `xvla` 和 `piper_sdk 0.6.1` 成功读取，程序退出码为 0。
- 关节与夹爪反馈频率均为 200 Hz；状态码为 0、运动状态为 0，但控制模式为 1（CAN 命令控制模式）。
- 六个关节均返回了角度，读取过程中没有观察到机械臂运动。
- 六个电机反馈均为已使能；本程序没有发送使能命令，说明使能状态在连接前已经存在。
- 夹爪反馈为负且 `homed=false`，因此当前数值只是未校零的相对行程；夹爪传感器和驱动器均未报告错误。
- 程序记录 `piper_init=false`、`enable_command_sent=false`、`motion_command_sent=false`。
- 退出后没有残留 Piper、ACT、LeRobot 或 ROS 2 控制进程，CAN 保持 `ERROR-ACTIVE` 且无总线错误。

`outputs/024_arm_inventory/results.json` 保存完整一次性结果，但属于生成文件，不提交 Git。

## 为什么重新简化

原方案一次加入了连续采样、时间序列、驱动电压、温度、详细故障位和统计汇总，虽然工程上完整，但不利于当前学习。

现在先读懂并验证这条主线：

```text
ConnectPort(piper_init=False)
→ GetArmJointMsgs / GetArmGripperMsgs / GetArmStatus
→ 单位换算
→ 保存 results.json
→ DisconnectPort
```

速度、温度和详细故障留到确认一次反馈正确之后。

## 下一步

核对此前是否运行过 ACT 的 `robot.connect()` 或独立使能脚本。确认机械臂有可靠支撑、急停可用并读懂厂商流程后，才单独设计失能检查和夹爪回零；这两项都属于写入硬件状态的操作。
