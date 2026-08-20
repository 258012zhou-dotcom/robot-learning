# 实验 005：点质量 PID 闭环控制

## 实验目标

用一个一维点质量模型验证基础动力学、闭环反馈、PID 控制、控制限幅和积分限幅。该实验是纯软件仿真，不涉及真实电机或机器人硬件。

## 模型

控制误差为：

`error = target_position - position`

PID 根据当前误差、累计误差和误差变化生成控制量。控制量在本实验中视为作用在单位质量上的力：

`acceleration = control / mass`

仿真使用半隐式欧拉法，先更新速度，再使用新速度更新位置：

`velocity_next = velocity + acceleration * dt`

`position_next = position + velocity_next * dt`

## 安全边界

- 控制输出限制在正负 12 以内。
- 积分状态限制在正负 2 以内，减轻积分饱和。
- 所有参数在进入仿真前检查有效性。

这些措施适用于仿真教学，但真实机器人还必须具备低速启动、硬件急停、位置和速度限幅以及通信失效保护。

## 运行

在项目根目录执行：

```bash
conda activate robot_learning
./scripts/run_tests.sh
PYTHONPATH=src python experiments/005_pid_control/run.py
```

## 实际结果

- 目标位置：5.0。
- 最终位置：约 5.0363。
- 最终绝对误差：约 0.0363。
- 最大超调：约 0.0464。
- 进入并保持在正负 0.05 容差带的时间：2.21 秒。
- 最大绝对控制量：12.0，达到设定限幅。

生成结果：

- `outputs/005_pid_control/results.json`
- `outputs/005_pid_control/pid_response.png`
