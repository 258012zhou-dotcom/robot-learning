# 学习笔记索引

笔记保存稳定知识、项目证据和知识空白，不保存原始聊天过程。阅读顺序：先看阶段 0 的工具链，再看 Python 实验基础，最后进入系统、并发、C++ 与网络。

## 阶段 0：开发工具

- Linux 与 Shell：[命令行基础](concepts/linux-command-line.md) · [进程、设备与网络环境](concepts/linux-process-device-network.md)
- Git：[基础](concepts/git-basics.md) · [分支与合并](concepts/git-branch-basics.md) · [远程仓库](concepts/git-remote-basics.md)
- Conda 环境：[阶段 0 检查点](weekly/2026-08-15-stage0-checkpoint.md) · [`environment.yml`](../environment.yml)
- VS Code、pytest 与调试：[Python 工作流](concepts/vscode-python-workflow.md) · [测试、调试与可复现性](concepts/testing-debugging-reproducibility.md)
- Codex：[CLI 基础](concepts/codex-cli-basics.md)

## 阶段 1：Python 实验基础

- 工程结构：[Python 工程基础](concepts/python-engineering-foundations.md) · [可靠软件边界](concepts/python-reliable-boundaries.md)
- 数值计算与可视化：[NumPy 轨迹分析](concepts/numpy-trajectory-analysis.md)
- 数据结构与过滤：[位置过滤](concepts/data-structures-and-position-filtering.md)

## 阶段 1：数学基础

- 线性代数：[机器人学习中的线性代数](concepts/linear-algebra-for-robotics.md)
- 微积分与优化：[微积分、梯度与数值优化](concepts/calculus-and-numerical-optimization.md)
- 概率统计：[机器人学习中的概率与统计](concepts/probability-statistics-for-robotics.md)
- 机器人数学与控制：[坐标变换、刚体运动、动力学与控制](concepts/rigid-motion-dynamics-control.md)

以上内容已完成第一轮系统复习，但尚未全部通过独立练习验证，因此路线图暂不标记为已掌握。

## 阶段 1：机器学习基础

- 数据划分、标准化、Tensor、自动微分、训练循环和评价：[PyTorch 训练、验证与评价流程](concepts/pytorch-training-workflow.md)
- 综合验证：[实验 004：PyTorch 学习二维运动模型](../experiments/004_learned_dynamics/README.md)

## 系统、并发与通信

- 并发模型与线程数据流：[Python 并发基础](concepts/python-concurrency-basics.md)
- TCP 传感器消息：[网络与 TCP 传感器基础](concepts/network-tcp-sensor-basics.md)
- C++ 构建与测试：[C++ 与 CMake 基础](concepts/cpp-cmake-basics.md)

## 记录与实验

- [阶段 0 检查点](weekly/2026-08-15-stage0-checkpoint.md) · [阶段 0 完成小结](weekly/2026-08-15-stage0-complete.md)
- [实验 001：点机器人轨迹](../experiments/001_point_robot/README.md) · [实验 002：位置过滤](../experiments/002_position_filter/README.md) · [实验 003：传感器线程](../experiments/003_sensor_thread/README.md)
- [实验 004：PyTorch 学习二维运动模型](../experiments/004_learned_dynamics/README.md)
