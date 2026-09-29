# 实验 053：Diffusion Policy 前置的双路线二维任务

## 目标与原理

先建立一个真正需要绕障、且同一观察下有两种有效动作路线的**教学仿真**，再考虑训练扩散策略。点机器人观察为 `[x, y, goal_x, goal_y, obstacle_x, obstacle_y, obstacle_radius]`，位置单位米；动作 `[vx, vy]` 是速度指令，单位米/秒，每步 `dt=0.1` 秒。障碍是圆形，直线到目标被挡住；运动线段碰到障碍即失败，到目标 `0.08` 米内且未碰撞即成功，最多 80 步。

同一场景用相同初始观察分别采集“下绕”和“上绕”示范，路线标签只保留为数据审计元信息，不提供给策略作输入。这是为检查**整段动作的多模态与闭环执行**准备的数据，不是 Diffusion Policy 或 ACT 论文复现。

## 数据与验证协议

场景 seed 在不同划分间不重叠：训练 0–199，验证 10000–10039，测试 20000–20039；先按**场景**划分，再为每个场景生成两条路线，避免一条路线训练、另一条路线测试的泄漏。执行脚本检查全部专家轨迹到达、与圆障碍的线段净距为正、直达路径被挡、上下路线位于障碍两侧、相同场景初始观察一致；保存后还核对每个场景恰有两条 Episode 且 split 相同。数据复用项目已有的 TransitionDataset 格式，保留 Episode、步号、路线元信息和拆分 ID。

单元测试检查：场景采样可复现、线段中段碰撞不会漏判、直走应撞障碍、两条专家路线应从相同观察无碰撞到达、非法动作应拒绝、超时与重置正确，并通过 Gymnasium 的环境契约检查。任何检查失败，都意味着任务或数据协议不能用于下一步训练。

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_two_route_navigation.py tests/test_trajectory_dataset.py tests/test_action_chunk_dataset.py tests/test_gymnasium_rollout.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/053_diffusion_two_route_foundation/run.py
```

配置见 `configs/053_diffusion_two_route_foundation.json`。数据、逐 Episode 审计结果和示意图在 Git 忽略的 `outputs/053_diffusion_two_route_foundation/`，不提交模型权重或生成数据。

## 本次实际结果与限制

在当前配置下，280 个场景各产生上下两条成功专家轨迹，共 **560 条 Episode、20,713 条转移**；最小专家线段净距约 `0.2443 m`，所有场景的起点—终点直线均穿障碍（最大净距也为 `−0.2636 m`）。首个场景的两条轨迹初始观察完全相同，第一动作约为 `[0.684, −0.415]` 与 `[0.646, 0.472]` m/s。训练/验证/测试分别有 400/80/80 条 Episode。脚本运行成功。

环境也通过 Gymnasium 契约检查。进一步连同共享轨迹/片段/rollout 测试复验，共 `50 passed, 1 warning`；唯一警告是未注册为 `gymnasium.make` 环境而无法检查其他渲染模式，非运行失败。

## 第二道关：确定性片段 BC 基线

`train_bc.py` 使用同一数据中的**上下两条**示范训练片段 BC（Behavior Cloning，行为克隆）；路线 ID 不进入输入。输入是当前 7 维状态，输出未来 8 步二维速度，执行前 4 步后重新观察、重新预测。这样固定了后续低维扩散策略要对齐的预测/执行长度。训练集拟合观察归一化；验证集选最低片段 MSE 的模型；测试集只在模型固定后评价。模型训练种子 42，60 个训练 epoch，属于**单种子试运行**，不是稳定性结论。

新增的单元测试检查：同一场景的两条路线均进入各自 split、验证/测试数值不参与归一化；测试用的 3 步预测片段仅执行指定的 2 步前缀就重新规划。失败意味着数据遗漏/泄漏或执行时间表有误。实测命令：

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_sequence_behavior_cloning.py tests/test_two_route_navigation.py tests/test_action_chunk_dataset.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/053_diffusion_two_route_foundation/train_bc.py
```

首条命令结果 `38 passed, 1 warning`；警告仍是 Gymnasium 未注册环境的其他渲染模式检查。训练结果：验证集最低片段 MSE 在第 56 个 epoch，约 `0.00759`；测试片段 MSE 约 `0.00770 (m/s)^2`，首动作 MSE 约 `0.00779 (m/s)^2`。闭环验证集和测试集各成功 **37/40 = 92.5%**；测试失败的 3 个场景（seed 20004、20013、20014）全部撞障碍，无超时。失败场景第一步纵向速度仅约 `−0.014`、`−0.009`、`−0.020 m/s`，明显小于专家示范的约 `±0.4 m/s`；但多数场景后来能转入某一侧路线，故“初始动作折中”不等于“整条轨迹必然失败”。结果与逐场景记录在 Git 忽略的 `outputs/053_diffusion_two_route_foundation/bc_results.json`，权重在同目录的 `bc_best_model.pt`。

## 第三道关：低维扩散策略试运行

`train_dp.py` 用**同一份上下路线数据**训练观察条件下的动作片段扩散模型：先给示范的 8 步二维动作加噪，网络依据当前观察、带噪片段和噪声步数预测所加噪声；推理时从随机片段反向去噪，执行前 4 步再观察、重采样。噪声日程是 32 步 cosine/DDPM，模型是小型 MLP，只是教学版低维机制，不含原论文的图像编码器与完整实现。路线 ID 仍不作为输入。训练集拟合归一化，固定噪声的验证集去噪 MSE 选择检查点，测试集仅在选定后评价。

新增单元测试分别用已知数值核对加噪公式、填充掩码、固定种子的采样与动作边界、执行前缀重规划、权重与归一化的保存/加载。失败意味着去噪机制、执行协议或复现性有误，不应继续解释闭环结果。实际运行：

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_lowdim_action_diffusion.py tests/test_sequence_behavior_cloning.py tests/test_two_route_navigation.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/053_diffusion_two_route_foundation/train_dp.py
```

测试为 `40 passed, 1 warning`；警告仍是未注册环境的渲染模式检查。单训练种子 42、80 epoch，最佳检查点在第 78 epoch。固定噪声评估下，验证去噪 MSE 从初始化约 `0.995` 降至 `0.0473`，测试约 `0.0478`。**这是噪声预测误差，不是动作 MSE，不能和 BC 的 `0.0077` 比大小。**

在验证场景 seed 10000 的**同一个初始观察**上独立生成 64 个片段，以前 4 步平均纵向速度的 `±0.1 m/s` 作诊断阈值：20 个偏上、41 个偏下、3 个接近中性；首动作纵向速度标准差约 `0.400 m/s`，强正负方向在同一前 4 步内切换的片段为 0。这说明这个场景的生成前缀有两类方向，但不能据此断定整条路线安全或已达到目标。

每个验证/测试场景预先固定做 4 次独立采样，**按每次 rollout 计分，不挑最好的那次**。验证为 `137/160` 成功、16 次碰撞、7 次超时；测试也为 `137/160` 成功、16 次碰撞、7 次超时。测试集每场景第 1 次采样为 `37/40`，与确定性 BC 的 `37/40` 相同；验证第 1 次采样为 `33/40`。这次低维扩散策略展示了不同路线采样，但**没有展示更好的闭环成功率**，而且有重采样后的碰撞与超时。逐次结果与配置保存在 Git 忽略的 `outputs/053_diffusion_two_route_foundation/dp_results.json` 与 `dp_best_model.pt`。

## 第四道关：独立训练种子与失败轨迹

保持数据哈希、网络、训练预算、预测 8/执行 4 步、32 步扩散和场景划分不变；在已有训练种子 42 外，固定补跑 123、777。不同种子的检查点和报告分开保存，不覆盖原始结果。每个模型仍以验证去噪 MSE 选检查点，测试均为 40 场景 × 4 次独立采样，**不是四次里挑最好的**。命令：

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/053_diffusion_two_route_foundation/train_dp.py --seed 123
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/053_diffusion_two_route_foundation/train_dp.py --seed 777
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/053_diffusion_two_route_foundation/analyze_dp_failures.py
```

| 训练种子 | 测试去噪 MSE | 测试成功/160 | 碰撞 | 超时 | 每场景第 1 次成功/40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.0478 | 137 | 16 | 7 | 37 |
| 123 | 0.0448 | 124 | 9 | 27 | 28 |
| 777 | 0.0449 | 111 | 23 | 26 | 27 |

去噪 MSE 略低的另外两个模型，闭环成功反而更少；3 个训练种子的成功数范围是 **111–137/160**，不能把 seed 42 的 `37/40` 当成稳定水平。480 条 rollout 反复使用同一批 40 个测试场景，不是 480 个独立任务。该补充是在看过首轮测试结果后开展的**探索性复验**，不是全新盲测；三模型预算与数据一致，但仍不足以推断算法一般性能，更不能和只训练一个种子的 BC 做公平算法排名。

`analyze_dp_failures.py` 从保存的检查点、场景 seed 与采样 seed 重放了 **480/480** 条测试轨迹，逐条核对首动作、步数、成功/碰撞/超时和最终距离，均一致。48 次碰撞都发生在首次 4 步重新规划之后（第 10–18 步），但这**不能单独证明**重新规划是原因。作为事后诊断，在圆障碍左边界之前先忽略平均纵向速度绝对值不超过 `0.1 m/s` 的片段，再检查相邻的强方向片段是否正负反转：碰撞中 43/48 出现，成功中 18/372 出现；它与碰撞相关，但阈值为事后诊断，未做因果干预。

60 次超时全部越过目标的 x 坐标，仍未进入半径 `0.08 m` 的成功区；按训练种子划分，超时轨迹曾达到的最近目标距离平均约 `0.088`、`0.088`、`0.105 m`，之后通常继续向右走。图 `outputs/053_diffusion_two_route_foundation/dp_failure_examples.png` 展示 seed 42 的一个碰撞和一个超时；逐轨迹诊断保存在 `dp_failure_analysis.json`。这说明失效不仅有“绕障选边”，还有目标附近没有可靠停下的问题。

这个二维环境仍是速度直接作用于位置的点机器人，没有接触动力学、相机、真实延迟或复杂物体操作。两条路线由脚本专家按预定路点生成；本实验是教学仿真，不是完整论文复现，也没有真实机器人证据。BC 与扩散模型的参数量、训练 epoch 和损失不同，数字不能作算法优劣排名；不依据测试集继续调任务或选择性报告种子。
