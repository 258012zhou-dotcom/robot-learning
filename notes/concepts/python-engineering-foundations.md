# Python 工程基础

## 核心概念

- **函数（function）**：接收输入并返回结果。`simulate_constant_velocity(...)` 把位置、速度和时间参数转换为轨迹。
- **类（class）**：把相关的数据和行为放在一起。`PositionBuffer` 保存最近的位置，并提供 `append()` 和 `mean()`。
- **模块（module）**：一个 Python 文件，例如 `point_robot.py` 或 `position_buffer.py`。
- **包（package）**：包含 `__init__.py` 的目录。`src/robot_learning/` 是项目包，因此可以用 `from robot_learning... import ...` 导入模块。

项目目录按职责划分：

| 目录 | 职责 |
| --- | --- |
| `src/` | 可复用的核心逻辑，不依赖某个实验目录 |
| `experiments/` | 单个实验的运行入口、说明和反思 |
| `tests/` | 自动验证核心逻辑和错误处理 |
| `configs/` | 可修改的实验参数，避免散落在代码中 |
| `outputs/` | 由代码生成的图片、日志和结果，可重新生成 |

## 项目中的实际证据

- 实验 001 的 `run.py` 从 `src/robot_learning/point_robot.py` 导入仿真和轨迹分析函数。
- 实验 002 的 `run.py` 复用 `PositionBuffer`，而没有在实验脚本中重新写滑动平均。
- `PointRobotConfig` 使用 `@dataclass(frozen=True)` 表示字段固定的配置对象；它把 JSON 字典转换为经过检查的数据。
- `tests/` 独立验证 `PointRobotConfig`、轨迹函数和 `PositionBuffer`，实验脚本只负责组织一次运行。

## 常见错误

- 直接运行脚本时找不到 `robot_learning`：需要按实验 README 使用 `PYTHONPATH=src`，或通过 pytest 的项目配置运行测试。
- 把通用算法复制到每个 `run.py`：后续修复时容易漏改；应放入 `src/` 后复用。
- 把实验参数硬编码到多个位置：改参数时容易不一致；应集中放入 `configs/`。
- 把图片、日志当作唯一证据：它们应能由固定代码和配置重新生成。

## 复习问题

1. 为什么 `PositionBuffer` 适合放在 `src/`，而不是放在实验 002 的 `run.py`？
2. `tests/` 与 `experiments/` 的职责分别是什么？
3. 当同一算法被两个实验使用时，应该如何安排代码位置？
4. `frozen=True` 解决的是哪类配置修改风险？
