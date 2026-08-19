# 实验 001：二维点机器人轨迹仿真

## 问题

本实验把机器人简化为二维平面上的一个点。已知机器人的初始位置、恒定速度、时间步长和仿真步数，计算并绘制它随时间变化的轨迹。

这个模型不考虑机器人的大小、朝向、加速度或碰撞，是学习机器人状态更新和离散时间仿真（Discrete-time Simulation）的基础实验。

## 原理与公式

设第 `k` 步的位置为：

```text
p_k = [x_k, y_k]
```

恒定速度为 `v = [v_x, v_y]`，时间步长为 `dt`，则下一步位置是：

```text
p_(k+1) = p_k + v * dt
```

从初始位置 `p_0` 出发，第 `k` 步也可以直接计算为：

```text
p_k = p_0 + k * dt * v
```

仿真执行 `steps` 次更新，但轨迹还要包含更新前的初始位置，因此返回数组的形状是：

```text
(steps + 1, 2)
```

## 本实验参数

| 参数 | 数值 | 含义 |
| --- | --- | --- |
| 初始位置 | `[0, 0]` | 从二维坐标原点出发 |
| 速度 | `[1, 0.5]` | x、y 方向每秒分别移动 1 和 0.5 |
| `dt` | `0.1` 秒 | 每次更新间隔 |
| `steps` | `100` | 更新 100 次 |

总仿真时间为 `0.1 × 100 = 10` 秒，所以预期最终位置为：

```text
[0, 0] + [1, 0.5] * 10 = [10, 5]
```

## 运行方法

在项目根目录中先激活项目环境，然后运行测试：

```bash
conda activate robot_learning
./scripts/run_tests.sh
```

测试用于检查原有的一维位置更新、二维轨迹形状、初始位置、最终位置和非法输入处理。

运行实验并生成轨迹图：

```bash
PYTHONPATH=src python experiments/001_point_robot/run.py
```

`PYTHONPATH=src` 让 Python 能从项目的 `src/` 目录中找到 `robot_learning` 包。程序会把图片保存到：

```text
outputs/001_point_robot/trajectory.png
```

## 预期结果

- 轨迹数组形状为 `(101, 2)`。
- 第一行是初始位置 `[0, 0]`。
- 最后一行约为 `[10, 5]`。
- 图像是一条从 `(0, 0)` 到 `(10, 5)` 的直线，因为两个方向的速度都保持不变。

## 实验配置与复现

运行参数位于 `configs/001_point_robot.json`，包括初始位置、速度、时间步长、步数和随机种子。

运行命令：

```bash
PYTHONPATH=src python experiments/001_point_robot/run.py
```

运行后应重新生成 `results.json`、`trajectory.png` 和 `run.log`。这些文件位于 `outputs/`，不提交到 Git。
