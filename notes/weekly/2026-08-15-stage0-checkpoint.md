# 阶段 0 检查点：开发环境与基础测试

当时已验证：项目位于 `~/AI_Project/robot-learning`；使用 `robot_learning` Conda 环境；Python、NumPy、Matplotlib、pytest 可用；最初的一维位置函数和 2 个测试通过。

已建立 `environment.yml` 与 `scripts/run_tests.sh`，后者要求在正确 Conda 环境中运行。

故障记录：ROS 2 自动加载可能干扰普通 Python 环境。普通学习使用 `robot_learning`；需要 ROS 2 时在专门终端手动 `source /opt/ros/humble/setup.bash`。

后续补齐：Linux 文件操作、VS Code、pytest、日志、配置、随机种子与可复现流程。
