# 实验 010：语义 Mask 到目标点云

## 目标

使用针孔相机模型，把合成 Z-depth 图像反投影为相机光学坐标系点云，并使用已知语义 Mask 分别提取正方形与圆形的三维点。

## 场景

- 背景平面约为 4.0 m。
- 正方形平面约为 1.5 m。
- 圆形平面约为 2.2 m。
- 深度包含高斯噪声、零值孔洞、`NaN` 和 `Inf`。
- Mask 类别为背景 0、正方形 1、圆形 2。

为了隔离三维几何问题，本实验使用已知正确的语义 Mask，不把实验 009 的模型误差混入当前分析。

## 核心公式

```text
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
Z = depth[v, u]
```

输出点位于相机光学坐标系：X 向右、Y 向下、Z 向前，单位为米。

## 验证

- 单元测试验证投影、反投影、单位转换与深度过滤。
- 运行时验证目标点云非空。
- 正方形中心深度应小于圆形，圆形应小于背景。
- 三维点重新投影到来源像素后的最大误差应不超过 `1e-4` 像素。
- 人工检查二维场景与三维散点图是否对应。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_depth_point_cloud.py
PYTHONPATH=src python experiments/010_depth_point_cloud/run.py
python -m json.tool outputs/010_depth_point_cloud/results.json
```

## 输出

- `outputs/010_depth_point_cloud/results.json`
- `outputs/010_depth_point_cloud/scene_overview.png`
- `outputs/010_depth_point_cloud/point_clouds.png`
- `outputs/010_depth_point_cloud/target_point_clouds.npz`
- `outputs/010_depth_point_cloud/run.log`

## 实际结果

使用固定种子 `42` 运行：

- 图像大小为 `120×160`，相机内参为 `fx=fy=140`、`cx=79.5`、`cy=59.5`。
- 过滤无效深度后，完整点云包含 18628 个点。
- 正方形点云包含 2325 个点，中心约为 `(-0.3751, -0.1283, 1.4999)` m。
- 圆形点云包含 1737 个点，中心约为 `(0.6362, 0.1961, 2.2006)` m。
- 三维点投影回来源像素的最大误差约为 `8.53×10⁻⁶` pixels。

二维深度图与三维散点图均通过人工检查：正方形位于相机左上侧且最近，圆形位于右下侧且稍远，背景平面约为 4 m；零值、`NaN`、`Inf` 深度孔洞没有进入点云。

## 当前边界

这是已配准合成 RGB-D 数据上的几何实验，不包含真实传感器标定、时间同步、畸变、点云坐标变换、降采样、离群点过滤或实例识别。
