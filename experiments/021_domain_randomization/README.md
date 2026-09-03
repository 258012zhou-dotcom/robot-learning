# 实验 021：动力学领域随机化与独立评估

## 目标

在每条 MuJoCo Episode 开始时改变机器人质量和关节阻尼，比较“只在 nominal 环境选择的 PD 控制器”和“在随机化范围内选择的 PD 控制器”面对不同动力学分布时的表现。

这里进行的是离散候选控制器调参，不是强化学习训练。它用于隔离领域分布、选择目标和独立评估的基本实验逻辑。

## 领域参数

| 分布 | 质量 | 阻尼 | 用途 |
| --- | --- | --- | --- |
| nominal | `1.0` | `0.4` | 固定标准环境 |
| randomized | `[0.5, 1.5]` | `[0.1, 0.8]` | 随机化选择与 IID 评估 |
| OOD | `[1.6, 2.5]` | `[0.01, 0.08]` | 更重、低阻尼的范围外评估 |

质量和阻尼在 `reset()` 时设置，并在整条 Episode 内保持不变。下次未指定参数的 reset 会恢复 nominal，避免参数泄漏。

## 控制器选择

候选网格：

```text
Kp ∈ {0.25, 0.5, 1.0, 2.0, 3.0}
Kd ∈ {0.0, 0.25, 0.5, 1.0, 2.0}
```

每个候选在 12 条选择 Episode 上评估。分数越低越好：

```text
100 × failure_rate
+ mean_duration
+ 5 × mean_overshoot
+ mean_final_distance
```

nominal 和 randomized 分别独立选择控制器。最终评估使用新的 task seed 和 domain seed，不能反过来修改候选或评分。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh \
  tests/test_point_robot_reach_env.py \
  tests/test_gymnasium_rollout.py
PYTHONPATH=src python experiments/021_domain_randomization/run.py
python -m json.tool outputs/021_domain_randomization/results.json
```

## 输出

- `controller_selection.csv`：两个选择分布上的全部候选和分数。
- `evaluation_episodes.csv`：每次评估的质量、阻尼、seed 与结果。
- `results.json`：被选控制器、领域定义和聚合结果。
- `domain_evaluation.png`：三种评估分布上的成功率、时间和超调。
- `ood_example.png`：相同 OOD 参数与目标下的代表轨迹。
- `run.log`：运行摘要。

## 本次结果

控制器选择得到两组不同参数：

- nominal 选择：`Kp=2.0`、`Kd=2.0`。
- randomized 选择：`Kp=3.0`、`Kd=2.0`。

两组控制器在三种评估分布中都达到 `60/60` 成功，因此成功率已经饱和，不能单独用来判断谁更鲁棒。随机化选择的控制器在 nominal 和 in-distribution 环境中平均完成得更快，但超调也更大；到了更重、低阻尼的 OOD 环境，它平均耗时 `6.0950 s`、平均超调 `0.2892 m`，均差于 nominal 选择控制器的 `5.6792 s` 和 `0.1731 m`。

这是一项有效的负结果：当前随机范围和选择分数鼓励了更激进的比例增益，却没有带来 OOD 性能提升。它说明领域随机化的效果取决于随机参数、范围、优化目标和评估分布，而不是只要加入随机化就会自动获得鲁棒性。

## 关键实验原则

- 任务 seed、领域 seed 和策略参数分开管理。
- 物理参数按 Episode 随机，而不是每个 step 突变。
- 训练范围内的新 seed 属于 in-distribution，不是 OOD。
- OOD 数据只能用于最终评价，不能用于挑选控制器。
- 领域随机化可能提高鲁棒性，也可能因为范围或评分设计不当产生负结果。
