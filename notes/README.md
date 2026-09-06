# 学习笔记索引

这里是复习入口，不是从头重学的任务清单。先选当前主题，读“快速复习”，有疑惑再看原理和小例子；实验 README 用来运行，reflection 用来查结果与局限。

## 怎么看最省时间

| 现在想解决的问题 | 从哪里看 |
| --- | --- |
| 数组维度、轨迹误差看不懂 | [NumPy 轨迹分析](concepts/numpy-trajectory-analysis.md) |
| 不确定该检查什么、测试说明了什么 | [可靠输入边界](concepts/python-reliable-boundaries.md) · [测试与可复现性](concepts/testing-debugging-reproducibility.md) |
| 机器人收到了命令却行为不对 | [TF2](concepts/ros2-tf2.md) · [控制](concepts/rigid-motion-dynamics-control.md) · [安全链路](concepts/ros2-robot-safety-basics.md) |
| 模型训练好了，换进程却不能用 | [训练与评价](concepts/pytorch-training-workflow.md) · [预训练与微调](concepts/pretraining-and-finetuning.md) · [LoRA](concepts/parameter-efficient-finetuning.md) |
| 准备学策略训练 | [环境接口](concepts/gymnasium-environment-and-rollout.md) · [轨迹数据](concepts/simulation-trajectory-data-collection.md) · [当前路线](../roadmap.md) |

公式默认用可直接阅读的文本：`@` 是矩阵乘法，`×` 是乘法，`abs(x)` 是绝对值，`Σ` 表示求和，`dt` 是时间间隔。每篇会解释本地变量含义；代码里的反斜杠可能是命令换行符，不应该一概删除。复杂数学如使用 `$...$`，需在 Markdown **预览**中阅读，而不是把源码转义符当公式。

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

数学已完成第一轮复习与部分应用：轨迹/变换、PyTorch 自动微分、PID 和一维 Kalman Filter 均有对应实验。复杂动力学、多维估计、数值优化理论仍需按任务补深，不把这些基础实验外推为全部数学掌握。

## 阶段 1：机器学习基础

- 数据划分、标准化、Tensor、自动微分、训练循环和评价：[PyTorch 训练、验证与评价流程](concepts/pytorch-training-workflow.md)
- 综合验证：[实验 004：PyTorch 学习二维运动模型](../experiments/004_learned_dynamics/README.md)

## 系统、并发与通信

- 并发模型与线程数据流：[Python 并发基础](concepts/python-concurrency-basics.md)
- TCP 传感器消息：[网络与 TCP 传感器基础](concepts/network-tcp-sensor-basics.md)
- C++ 构建与测试：[C++ 与 CMake 基础](concepts/cpp-cmake-basics.md)

## 阶段 2：ROS 2 机器人软件

- 工作空间、功能包、环境叠加与 colcon：[ROS 2 工作空间与功能包](concepts/ros2-workspace-and-package.md)
- 节点配置、启动参数与动态更新：[ROS 2 Parameter](concepts/ros2-parameters.md)
- 多节点启动、参数连接与进程管理：[ROS 2 Launch](concepts/ros2-launch.md)
- 自动启动节点并验证真实 Topic 通信：[ROS 2 最小集成测试](concepts/ros2-integration-testing.md)
- Topic 数据的记录、检查、回放与复现边界：[ROS 2 rosbag](concepts/ros2-rosbag.md)
- 坐标树、静态与动态变换、旋转和平移：[ROS 2 TF2](concepts/ros2-tf2.md)
- Link、Joint、几何、惯量与模型发布：[ROS 2 URDF](concepts/ros2-urdf.md)
- Fixed Frame、RobotModel、TF 与配置复现：[ROS 2 RViz](concepts/ros2-rviz.md)
- 关节空间、任务空间、正运动学与简单逆运动学：[机器人运动学基础](concepts/robot-kinematics-basics.md)
- P、PI、PD、稳定性、限幅与 ROS 2 闭环控制：[动力学与 PID 控制](concepts/rigid-motion-dynamics-control.md)
- 累计编码器计数、差速轮运动模型、`/odom` 与 TF：[ROS 2 编码器与差速轮里程计](concepts/ros2-wheel-encoder-odometry.md)
- Image、CameraInfo、LaserScan、光学坐标系与传感器 QoS：[ROS 2 摄像头与二维激光雷达](concepts/ros2-camera-lidar-basics.md)
- Prediction、Correction、Kalman Gain 与 P/Q/R：[状态估计与一维 Kalman Filter](concepts/state-estimation-kalman-basics.md)
- 指令限幅、Watchdog、软件急停、非法输入停车与仿真安全运动链路：[ROS 2 机器人安全基础](concepts/ros2-robot-safety-basics.md)
- 持续扩展项目：[ROS 2 Point Robot Workspace](../projects/ros2_point_robot_ws/README.md)

## 阶段 3：视觉、语言与多模态基础

- 图像契约、Letterbox、坐标与内参同步、HSV 基线和模型 Tensor：[具身智能视觉输入与预处理](concepts/vision-input-preprocessing.md)
- 卷积、分类训练、混淆矩阵、IID/OOD 与领域随机化：[CNN 图像分类与泛化](concepts/cnn-classification-and-generalization.md)
- 边界框、IoU、联合损失、置信度过滤、NMS 与检测评价：[目标检测基础](concepts/object-detection-foundations.md)
- 像素 Mask、Encoder-Decoder、Skip Connection、类别不平衡与 mIoU：[语义分割基础](concepts/semantic-segmentation-foundations.md)
- Z-depth、相机内参、反投影、语义点云与坐标系：[深度图与点云基础](concepts/depth-and-point-cloud-foundations.md)
- Encoder、Embedding、对比学习、表示坍塌与冻结评价：[视觉表示学习](concepts/visual-representation-learning.md)
- Token、Patch、Q/K/V、Multi-Head、Mask、Transformer Block 与位置编码：[Transformer、Attention 与 Tokenization](concepts/transformer-attention-and-tokenization.md)
- Checkpoint、Linear Probe、分层学习率、Full Fine-tuning 与灾难性遗忘：[预训练、微调与迁移评价](concepts/pretraining-and-finetuning.md)
- 双编码器、图文对比学习、语义检索、假负样本与 Zero-shot 边界：[视觉语言模型基础](concepts/vision-language-model-basics.md)
- PEFT、LoRA、Rank/Alpha、模块注入、权重合并与框架快速路径：[参数高效微调与 LoRA](concepts/parameter-efficient-finetuning.md)
- 表示、对齐、融合、Cross-Attention、Grounding 与单模态消融：[多模态融合与语言条件视觉定位](concepts/multimodal-fusion-and-grounding.md)
- 综合验证：[实验 006：视觉预处理与坐标同步](../experiments/006_vision_preprocessing/README.md)
- 综合验证：[实验 007：CNN 合成图像分类](../experiments/007_cnn_classification/README.md)
- 综合验证：[实验 008：单目标图像检测](../experiments/008_single_object_detection/README.md)
- 综合验证：[实验 009：合成图像语义分割](../experiments/009_semantic_segmentation/README.md)
- 综合验证：[实验 010：语义 Mask 到目标点云](../experiments/010_depth_point_cloud/README.md)
- 综合验证：[实验 011：监督与对比视觉表示](../experiments/011_visual_representation/README.md)
- 综合验证：[实验 012：位置编码与 Token 顺序](../experiments/012_transformer_token_order/README.md)
- 综合验证：[实验 013：视觉预训练与小样本微调](../experiments/013_visual_transfer_learning/README.md)
- 综合验证：[实验 014：视觉语言表示对齐](../experiments/014_vision_language_alignment/README.md)
- 综合验证：[实验 015：LoRA 语言别名适配](../experiments/015_lora_language_adaptation/README.md)
- 综合验证：[实验 016：语言条件视觉目标定位](../experiments/016_multimodal_grounding/README.md)

## 阶段 4：仿真与机器人学习

- MJCF、`MjModel`、`MjData`、`qpos/qvel/ctrl` 与 `mj_step`：[MuJoCo 模型、状态与物理步进](concepts/mujoco-model-data-step.md)
- Observation、Action、Reward、Episode、输入验证与软限位边界、策略评估：[Gymnasium 环境、Episode 与策略评估](concepts/gymnasium-environment-and-rollout.md)
- 物理频率、控制周期、`frame_skip`、动作语义与 Reward 时间尺度：[控制频率与动作表示](concepts/control-frequency-and-action-representation.md)
- Transition、Episode 边界、数据划分、采集分布与可复现存储：[仿真轨迹数据采集](concepts/simulation-trajectory-data-collection.md)
- 领域参数、Episode 随机化、nominal/IID/OOD、评分权衡与鲁棒评估：[领域随机化与鲁棒评估](concepts/domain-randomization-and-robust-evaluation.md)
- Reality Gap、执行与观测失配、系统辨识、可辨识性、数据隔离与安全迁移：[Sim-to-Real 与系统校准基础](concepts/sim-to-real-calibration-foundations.md)
- 批量接口、同步与异步 worker、吞吐量、开销和公平基准：[向量环境与训练性能](concepts/vectorized-environments-and-training-performance.md)
- 综合验证：[实验 017：MuJoCo 最小物理步进](../experiments/017_mujoco_step/README.md) · [实验 018：Gymnasium 到达任务与控制基线](../experiments/018_gymnasium_rollout/README.md) · [实验 019：控制频率对闭环行为的影响](../experiments/019_control_frequency/README.md) · [实验 020：可复现的仿真轨迹数据集](../experiments/020_simulation_dataset/README.md) · [实验 021：动力学领域随机化与独立评估](../experiments/021_domain_randomization/README.md) · [实验 022：Sim-to-Sim 失配校准与迁移评价](../experiments/022_sim_to_sim_calibration/README.md) · [实验 023：向量环境性能基准](../experiments/023_vector_environment_benchmark/README.md)
- 阶段总结：[阶段 4 完成小结](weekly/2026-09-05-stage4-complete.md)

## 早期实验、总结与故障排查

- [阶段 0 检查点](weekly/2026-08-15-stage0-checkpoint.md) · [阶段 0 完成小结](weekly/2026-08-15-stage0-complete.md)
- [001 点机器人轨迹](../experiments/001_point_robot/README.md) · [002 位置过滤](../experiments/002_position_filter/README.md) · [003 传感器线程](../experiments/003_sensor_thread/README.md) · [005 PID](../experiments/005_pid_control/README.md)
- 其余实验入口已放在所属阶段，不重复堆叠；学习顺序和待办以 [roadmap.md](../roadmap.md) 为准。

- [ROS 2 在 WSL 与 TUN 环境中的 DDS 发现问题](troubleshooting/ros2-dds-wsl-tun.md)
