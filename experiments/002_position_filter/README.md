# 实验 002：二维位置滑动平均过滤

## 实验目标

模拟一个静止在二维平面中的机器人。传感器对其位置进行 100 次测量，但每次测量都含有二维高斯噪声（Gaussian Noise）。实验使用 `PositionBuffer` 保存最近几次观测，并计算它们的平均值作为过滤后的位置估计。

## 滑动平均原理

若最近 `N` 次位置观测为 `z_1, z_2, ..., z_N`，滑动平均估计为：

```text
position_estimate = (z_1 + z_2 + ... + z_N) / N
```

随机噪声有正有负，平均后会部分抵消，因此对于静止目标，估计通常比单次原始观测更接近真实位置。

## 配置

配置文件是 `configs/002_position_filter.json`：

- `true_position`：机器人的固定真实二维位置。
- `observation_count`：传感器观测次数。
- `noise_std`：每个坐标轴上的高斯噪声标准差。
- `buffer_capacity`：滑动平均使用的最近观测数量。
- `seed`：随机种子，保证每次运行得到相同的随机观测。

## 运行命令

在项目根目录运行：

```bash
conda activate robot_learning
./scripts/run_tests.sh
PYTHONPATH=src python experiments/002_position_filter/run.py
```

## 输出

- `outputs/002_position_filter/results.json`：配置摘要、原始观测 RMSE 和过滤后 RMSE。
- `outputs/002_position_filter/filter_result.png`：原始误差与过滤误差随观测时间变化的对比图。

二维 RMSE（Root Mean Squared Error，均方根误差）衡量每一步估计相对真实位置的整体误差；数值越小，表示估计越准确。
