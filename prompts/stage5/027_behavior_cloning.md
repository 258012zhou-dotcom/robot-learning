# 实验 027：Behavior Cloning 提示词

## 提示词开始

你正在 `/home/zxd/AI_Project/robot-learning` 项目中开展阶段 5A 的实验 027。请先完整阅读根目录 `AGENTS.md`，并检查当前 Git 状态；保留用户已有的 `.vscode/settings.json` 修改和所有无关改动。随后自主完成下面的任务，不要只给计划，也不要操作任何真实硬件。

目标：复用项目已有的一维 MuJoCo 到达环境、P 控制专家和 Episode 轨迹数据，训练一个最小 Behavior Cloning（BC）策略，并用未见过的环境 seed 做闭环评价。这个实验要让我理解“离线动作误差较小”为什么不等于“闭环任务成功”。

论文阅读：使用 Brenna Argall、Sonia Chernova、Manuela Veloso 和 Brett Browning 的综述 [A Survey of Robot Learning from Demonstration](https://publications.ri.cmu.edu/a-survey-of-robot-learning-from-demonstration) 建立概念框架。无需逐页翻译；先阅读摘要、LfD 的问题分类、示范数据获取、策略推导和评价相关部分。结合当前项目回答并写入实验 README：示范者是谁、状态和动作空间是什么、策略怎样从示范得到、离线评价与任务级评价各是什么。论文原有范围很广，本实验只复现“状态到动作的监督策略 + 仿真闭环评价”这一基础机制，不声称复现整篇综述。

开始编码前，先用适合初学者的中文简要解释：

1. BC 如何把 `observation → action` 当作监督学习；输入、标签和损失各是什么。
2. 为什么按 Episode 划分数据，不能随机打散 Transition 后再划分。
3. 离线 MSE、闭环成功率、最终距离和累计回报分别说明什么。
4. 本实验将修改哪些文件，怎样验证；若新增测试，先说明测试类型、对象、输入、期望结果和失败含义。

实现范围：

- 阅读并复用 `experiments/018_gymnasium_rollout/`、`experiments/020_simulation_dataset/`、`src/robot_learning/point_robot_reach_env.py`、`src/robot_learning/trajectory_dataset.py` 及相关测试，不复制已有环境和数据采集逻辑。
- 创建 `experiments/027_behavior_cloning/README.md`、`reflection.md` 和运行入口；可复用的模型、训练、保存加载和评价逻辑放入 `src/robot_learning/`，测试放入根目录 `tests/`，配置放入 `configs/`。
- 训练数据只使用 P 控制专家轨迹。随机策略数据只作为已有数据分布背景，不能混入专家动作标签。
- 如果 `outputs/020_simulation_dataset/dataset.npz` 不存在，使用已有实验入口重新生成，不手工伪造数据。
- 严格按现有 Episode split 使用 train、validation、test；归一化统计只允许由 train split 计算，并随模型一起保存。
- 使用简单、可读的小型 MLP 输出连续动作；动作发送到环境前遵守环境 Action Space。不要引入大型框架或复杂网络。
- 固定并记录 Python、NumPy、PyTorch、环境和评价 seed。保存可独立加载的模型、模型结构所需配置与归一化参数；新进程加载后无需重新训练即可评价。
- 闭环评价必须使用训练和调参时未见过的环境 seed，并在相同 seed 上比较：随机策略、P 控制器和 BC。
- 主要指标至少包括成功率、最终距离、累计回报和 Episode 步数；另报告 test split 的动作 MSE。不要用 test 指标选超参数。
- 输出逐 Episode 原始结果和聚合结果，保留至少一条同目标下的策略轨迹图；清楚标注这是仿真结果。
- 当前实验只建立 BC 基线，不加入 DAgger、PPO、图像输入、Action Chunk、领域随机化或真实机器人部署，这些属于后续模块。

验证要求：

- 为数据选择、train-only 归一化、动作 shape/范围、模型保存加载一致性，以及固定 seed 下闭环评价的关键契约编写必要测试。
- 运行相关既有测试和新增测试；训练一次并从保存的 checkpoint 在新进程中完成评价。
- 检查结果文件的字段和数值是本次真实运行生成的。没有运行的测试或评价不能写成通过。
- BC 不必超过 P 控制专家；如果结果较差，保留负结果并从数据覆盖、误差累积、归一化和动作饱和等方面分析，禁止为了“好看”修改成功标准。
- 更新实验 README 和 reflection，记录实际命令、环境、seed、数据量、最佳 checkpoint 选择依据、最终指标、失败类型和适用范围。
- 完成后运行 `git diff --check`。不要提交或推送 Git，除非我另行要求。

最终回复请先给出实验结论，再说明修改、实际验证、关键指标、限制以及我应该阅读的三个关键断言或代码段。若发现现有数据或环境契约存在问题，先用证据说明并做最小修复，不要绕过它。

## 提示词结束
