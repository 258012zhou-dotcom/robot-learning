# Python 工程基础

函数处理输入到输出；类把状态和相关操作放在一起；模块是一个 `.py` 文件；包是可导入的模块目录。`src/robot_learning/` 是本项目的包。

| 位置 | 职责 |
| --- | --- |
| `src/` | 可复用核心逻辑，如轨迹仿真和 `PositionBuffer` |
| `experiments/` | 单次实验的入口、说明和反思 |
| `tests/` | 自动验证正常行为和错误处理 |
| `configs/` | 集中管理可修改参数 |
| `outputs/` | 由运行重新生成的图像、日志和结果 |

项目证据：实验 001 从 `src` 导入轨迹函数；实验 002 复用 `PositionBuffer`；`PointRobotConfig` 用不可变数据类承接已验证配置。

避免把通用算法复制进每个 `run.py`，也不要把同一个参数散落在多处。直接运行实验时按 README 设置 `PYTHONPATH=src`，否则可能找不到包。
