# AGILEX PIPER 设备资料

本目录是未来在 PIPER 上部署策略时的快速入门资料：复用已经弄清的连接、接口、单位和设备特点，减少重复排查，加快从仿真到设备适配的准备。当前主线仍是仿真实验与论文复现，真机部署在用户决定开展时再进行。

## 未来部署时从哪里开始

1. 阅读本页的型号和历史环境，核对当时使用的软件与当前设备是否一致。
2. 查阅 [操作与工具](operation.md)，了解连接、反馈、使能、停止和退出的区别，并找到已有查询脚本。
3. 查阅 [已验证事实与已知问题](findings.md)，重点检查关节偏差、限位及 ACT 的单位和退出处理。
4. 按选定策略核对观察、动作顺序与单位、相机输入和控制频率；只补本次部署需要而尚未验证的部分。

已有工具覆盖反馈、参数查询和使能观察，并非完整策略部署程序。这样保留资料是为了复用已有经验，不能把历史验证范围之外的行为视为已完成验证。

- [操作与工具](operation.md)：预览入口、单位、停止和断开的区别。
- [已验证事实与已知问题](findings.md)：区分实测、源码分析和现场信息。
- 历史 024：[设备盘点](records/inventory.md)、[只读反馈记录](records/inventory_reflection.md)。
- 历史 025：[参数查询](records/parameter_query.md)、[验证记录](records/query_reflection.md)、[首次使能说明](records/enable_observation.md)。

历史环境为 `can0`、1 Mbps、`piper_sdk 0.6.1`，单独从臂固件回复为 `S-V1.7-3`。型号为 AGILEX PIPER，不公开设备序列号或登录凭据。以上是当时的环境记录，未来使用前重新核对。

脚本已从旧实验目录迁至 `scripts/`，共享源码、测试和 `configs/024_arm_inventory.json` 保留原位；配置中的历史实验标识与 `outputs/024_arm_inventory`、`outputs/025_piper_safety_query` 保持不变，以便追溯旧结果。原 026 的文件当前本地不存在，未从远程重新获取或重建；已有对话证据归纳在 findings 中。

## 2026-09-20 迁移检查

使用本地 `robot_learning` 环境执行：

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_piper_telemetry.py tests/test_piper_safety_query.py tests/test_piper_firmware_capture.py tests/test_piper_enable_once.py
for entry in hardware/piper/scripts/*.py; do PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python "$entry" || exit; done
git diff --check
```

结果：22 项既有离线测试通过，4 个默认预览入口正常结束，差异格式检查通过；另检查 13 份相关 Markdown 的本地链接，无失效链接。本次未连接远程电脑或真机，未重跑历史硬件验证。
