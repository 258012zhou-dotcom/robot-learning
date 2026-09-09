# 实验 025：参数查询与首次使能反馈验证

**状态：已完成。** 2026-09-08 完成现场验证，2026-09-09 按本实验范围收尾；不要求补做故障停止测试。

## 完成依据与范围

- 完成固件及六轴角度、速度、加速度限制的查询代码、单位转换和异常处理。历史双臂限位数据保留来源限制，不作为已验证的单臂运动限位。
- 隔离主臂后读取 `S-V1.7-3`，并用原始 CAN 回复交叉核对；格式及回复核对不等于固件安全认证。
- 首次使能时六轴反馈由全部未使能变为已使能，模式仍为待机，检查的故障标志均为正常。用户现场未观察到明显运动或异响。
- 相关模拟测试曾运行得到 18 passed；模拟测试不替代实际硬件验证。

使能步骤与结果见 [使能反馈验证](enable_observation.md)，完整证据及历史问题见 [实验反思](reflection.md)。本实验不包含主动关节运动、失能测试、位置保持精度、急停、Watchdog 或通信中断保护验证。这些项目未验证、不影响 025 收尾，也不因此被认定安全；后续运动条件另行评估。

## 参数查询子步骤

目标：查询固件与六个关节的角度、速度、加速度限制。结果只是设备报告的参数，不代表已经验证其执行效果，也不是后续实验推荐速度。

流程：连接接收线程 → 发送三组查询 → 最多等待 3 秒 → 检查完整性 → 转换单位 → 断开 → 保存。

`SearchAllMotorMaxAngleSpd`、`SearchAllMotorMaxAccLimit` 和 `SearchPiperFirmwareVersion` 已按远程安装的 piper_sdk 0.6.1 源码检查。查询会发送 CAN 请求；没有使能、失能、重置、改限位或运动调用。

角度限制原始值每单位为 0.1°；速度为 0.001 rad/s；加速度为 0.001 rad/s²。注意角度限制的比例与实验 024 的当前位置反馈不同。

## 在远程实验电脑操作

先预览，不连接 CAN：

```bash
cd /media/zhao/F/zxd/robot-learning
PYTHONPATH=src /media/zhao/F/envs/xvla/bin/python experiments/025_piper_safety_query/run.py
```

现场确认机械臂静止、其他控制程序已停止后，进行一次查询：

```bash
timeout 15s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /media/zhao/F/envs/xvla/bin/python experiments/025_piper_safety_query/run.py --query-once
```

关节回复缺失会报错，不保存本次结果。六关节已完整回复但固件格式异常时，等待结束后保存 `query_status=partial`、`firmware_version=null` 和 SDK 原始返回值 `firmware_raw`。已有结果文件可能是旧运行留下的，不能把它当作失败运行的结果。

版本格式检查接受厂商使用的 `S-V1.7-3` 格式；其他格式一律留待核对，不证明设备不支持或固件损坏。即使格式通过，也只表示文本符合检查规则，不是厂商工具验证。

结果位于 `outputs/025_piper_safety_query/results.json`，不提交 Git。正常退出只断开本程序连接，不能用来停止其他程序已经发起的运动。

此前将末端按钮称为已确认的重置按钮，证据不足；其位置描述与示教按钮相符，不能据此认定为复位或急停。故障停止安全性未验证且不属于本实验验收范围。

## 离线验证

`tests/test_piper_safety_query.py` 使用模拟回复验证单位、缺失关节和异常断开，不连接 CAN。通过这些测试不能证明真机参数已经读到。
