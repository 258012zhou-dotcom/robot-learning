# 实验 017：MuJoCo 最小物理步进

## 目标

用一个单滑动关节点机器人理解 MuJoCo 的模型、状态、执行器和时间步进，不经过 Gymnasium 封装。

## 模型

- `x_slide`：沿世界 x 轴移动的单自由度滑动关节。
- `x_motor`：范围为 `[-1, 1]` 的电机控制输入。
- 质量：`1.0 kg`。
- 关节阻尼：`0.4`。
- 重力：关闭，使第一轮实验只研究一维驱动力和阻尼。
- 仿真步长：`0.01 s`。

控制量是驱动力性质的输入，不是目标位置。位置和速度由动力学及数值积分共同决定。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_mujoco_basics.py
PYTHONPATH=src python experiments/017_mujoco_step/run.py
PYTHONPATH=src python experiments/017_mujoco_step/view.py
python -m json.tool outputs/017_mujoco_step/results.json
```

## 输出

- `results.json`：模型维度和最终状态。
- `trajectory.csv`：每个仿真采样点的时间、位置、速度与控制量。
- `state_trajectory.png`：状态变化曲线。
- `run.log`：运行摘要。

## 实际结果

使用 `mujoco 3.12.0` 运行并通过 6 个针对模型和状态步进的单元测试：

- 模型维度：`nq=1`、`nv=1`、`nu=1`。
- 300 个 `0.01 s` 步长得到 `3.00 s` 仿真时间。
- 恒定控制 `u=1.0` 后，最终位置为 `3.1325 m`。
- 最终速度为 `1.7470 m/s`。
- 零控制测试中位置和速度保持为零。
- 使用新的 `MjData` 重复仿真得到完全一致的轨迹。
- MuJoCo passive Viewer 已在 WSLg 中正常打开，机器人能够按仿真状态沿 x 轴运动并正常退出。

模型满足一阶阻尼动力学 `m·dv/dt = u - b·v`。在 `m=1`、`u=1`、`b=0.4` 时，理论三秒速度为约 `1.7470 m/s`，与 MuJoCo 结果一致。
