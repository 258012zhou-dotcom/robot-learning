# PIPER 操作参考

## 快速复习

连接、使能、发送目标、停止和失能是不同操作。读取到使能状态不代表该读取程序发送了使能。退出程序也不等于停止机器人。

## 保留的设备工具

从仓库根目录运行以下默认预览，不连接 CAN：

```bash
PYTHONPATH=src python hardware/piper/scripts/read_once.py
PYTHONPATH=src python hardware/piper/scripts/query_parameters.py
python hardware/piper/scripts/capture_firmware.py
python hardware/piper/scripts/enable_once.py
```

`read_once.py --read-once` 接收反馈；`query_parameters.py --query-once` 和 `capture_firmware.py --capture` 会发送查询请求。`enable_once.py --execute` 会更改硬件使能状态，并要求现场输入确认，不能当作只读工具。历史操作详见 records，不要求重跑。

当前位置的原始单位为 0.001°；查询角度限位的原始单位为 0.1°；夹爪反馈单位为 0.001 mm。不能混用这些比例。ACT 输入语义还需按实际代码核对。

## 停止与退出

以下依据 2026-09-12 检查的 SDK 0.6.1 和现场 ACT 副本，不外推到所有固件版本：

| 接口或流程 | 代码行为 |
| --- | --- |
| `EmergencyStop(0x01)` | 发送快速停止指令；未在本次运动中验证效果 |
| `EmergencyStop()` | 默认参数 0，表示无效 |
| `DisconnectPort()` | 关闭接收和 CAN 连接，不发送停止或失能 |
| `ResetPiper()` / `EmergencyStop(0x02)` | 发送同一恢复指令；SDK 明确警告失电下落 |
| ACT `robot.disconnect()` | 先发送安全位目标，等待 5 秒，再失能 |

用户报告停止和通信断开后保持力矩；这是现场信息，不等于已测得断线瞬间停止或保持精度。设备没有独立物理急停按钮；末端按钮不能按急停使用。涉及新运动时按具体任务评估，不能复用 ACT 的退出流程来实现“保持当前位置”。

## 离线检查

```bash
PYTHONPATH=src python -m pytest -q tests/test_piper_telemetry.py tests/test_piper_safety_query.py tests/test_piper_firmware_capture.py tests/test_piper_enable_once.py
```

这些是假设备或纯计算测试，验证软件逻辑，不连接真机，不证明运动和停止效果。
