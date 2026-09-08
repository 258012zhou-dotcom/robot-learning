# 实验 025：安全参数查询（第一步）

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

用户确认实体按钮为重置按钮；它不是已验证的急停。保持位置、通信中断保护和紧急停止均待后续核对，本步不进行这些测试。

## 离线验证

`tests/test_piper_safety_query.py` 使用模拟回复验证单位、缺失关节和异常断开，不连接 CAN。通过这些测试不能证明真机参数已经读到。
