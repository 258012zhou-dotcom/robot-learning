# VS Code Python 工作流

先确认左下角是 `WSL: Ubuntu-22.04`，并选择 `robot_learning` Conda 解释器；否则编辑器和终端可能使用不同 Python 环境。

测试入口是烧杯图标，项目测试位于 `tests/`；完整测试命令是 `./scripts/run_tests.sh`。在代码行左侧设置红点断点，调试时查看变量，`F10` 单步执行但不进入函数。

本项目的测试环境禁用 pytest 自动插件加载，避免 ROS 插件干扰普通 Python 实验。
