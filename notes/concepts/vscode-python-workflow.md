# VS Code Python 工作流

先确认左下角是 `WSL: Ubuntu-22.04`，并选择 `robot_learning` Conda 解释器；否则编辑器和终端可能使用不同 Python 环境。

测试入口是烧杯图标，项目测试位于 `tests/`；完整测试命令是 `./scripts/run_tests.sh`。在代码行左侧设置红点断点，调试时查看变量，`F10` 单步执行但不进入函数。

本项目的测试环境禁用 pytest 自动插件加载，避免 ROS 插件干扰普通 Python 实验。

终端和 VS Code 必须使用同一 `robot_learning` 环境，否则“编辑器能导入、终端不能运行”或相反。遇到这种现象，先检查解释器选择、终端环境和 `pyproject.toml` 的测试路径。
