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

- [x] 创建项目目录
- [x] 初始化 Git 仓库
- [x] 配置 WSL Git
- [ ] 制定学习路线
- [ ] 完成第一个实验
- [ ] 建立测试和复盘习惯

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
