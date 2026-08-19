# Robot Learning

这是我的具身智能（Embodied AI）学习与实验项目。

近期目标是建立扎实的软件工程、机器学习和机器人系统基础，逐步具备具身智能软件工程师的能力；长期目标是能够阅读和复现论文、设计实验，并向研究型工程师或科学家方向发展。

## 学习目标

- 掌握 Linux、Git、Python、C++ 和软件工程基础
- 理解机器人运动学、控制、感知和 ROS 2 系统
- 掌握深度学习、计算机视觉和 Transformer 基础
- 学习机器人仿真、模仿学习和强化学习
- 理解机器人数据集、策略训练、评估和部署
- 逐步学习多模态模型和 Vision-Language-Action 模型
- 建立论文阅读、复现、消融实验和研究写作能力
- 形成可以展示、测试和复现的项目作品集

## 项目结构

- `notes/`：概念、课程笔记和故障排查记录
- `experiments/`：独立的小型学习实验
- `projects/`：较完整的阶段项目
- `src/`：可以复用的程序代码
- `configs/`：程序和机器人参数
- `tests/`：自动测试
- `scripts/`：运行、分析和辅助脚本
- `data/sample/`：可以提交的少量示例数据
- `data/local/`：不提交到 Git 的本地大型数据
- `outputs/`：程序生成的临时结果
- `references/`：课程资料、论文和参考链接

## 每个实验应包含

1. 实验目标
2. 必要的理论知识
3. 环境与依赖
4. 实现步骤
5. 运行方法
6. 验证方法
7. 实验结果
8. 问题与反思

## 当前状态

- [x] 建立 WSL Ubuntu 项目与 Conda 环境
- [x] 建立 Git 和 GitHub 工作流
- [x] 完成阶段 0：开发与科研工具
- [x] 完成阶段 1 的 Python 与软件工程第一轮学习和项目验证
- [ ] 阶段 1 数学基础已完成第一轮系统复习，待综合练习验证
- [ ] 阶段 1 机器学习基础已完成综合训练和误差分析，仍需验证过拟合与正则化

`[x]` 表示完成基础学习并至少实际验证一次，不代表熟练或精通。

## 当前实验与练习

- 实验 001：二维点机器人轨迹，验证二维匀速运动、轨迹统计和可视化。
- 实验 002：位置滑动平均过滤，验证随机传感器噪声下的滑动平均与 RMSE。
- 实验 003：传感器线程数据流，验证 `Thread`、`Queue`、`Event`、结束标记和异常传递。
- 实验 004：在 GPU 上训练 PyTorch 线性模型，验证数据划分、标准化、自动微分、最佳模型和基线评价。
- `projects/cpp_point_robot`：验证 C++17 点机器人函数、CMake 构建和 CTest。
- `projects/python_concurrency_demo`：验证多进程与 `asyncio` 的基本运行边界。
- `projects/tcp_sensor_demo`：验证本机 TCP 上的 JSON 位置样本与确认消息。

## 学习原则

先理解问题和原理，再编写代码。每次实验都应能够运行、验证并解释结果。

## 本地开发

首次创建项目环境：

```bash
conda env create -f environment.yml
conda activate robot_learning
```

运行项目测试：

```bash
./scripts/run_tests.sh
```

学习笔记索引见 [notes/README.md](notes/README.md)。
