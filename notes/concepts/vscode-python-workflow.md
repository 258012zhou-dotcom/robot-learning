# VS Code Python 工作流

- 左下角显示 `WSL: Ubuntu-22.04`，确认正在编辑 Ubuntu 中的项目。
- 为项目选择 `robot_learning` Conda 解释器。
- 测试入口是左侧烧杯，项目使用 pytest，测试目录是 `tests`。
- 运行全部测试：`./scripts/run_tests.sh`。
- 断点是代码行左侧的红点；调试单个测试时可以查看变量。
- `F10`：执行下一行，不进入函数内部。
- 本项目的 VS Code 测试环境禁用 pytest 自动插件加载，避免 ROS 插件干扰普通 Python 测试。