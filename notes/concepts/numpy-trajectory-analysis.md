# NumPy 轨迹分析

- 机器人轨迹通常表示为 `(T, D)`，其中 `T` 是时间点，`D` 是状态维度。
- `np.diff(trajectory, axis=0)` 计算相邻时间点之间的变化。
- `np.linalg.norm(vectors, axis=1)` 计算每个向量的长度。
- 位移是终点减起点；位移距离是起点到终点的直线距离。
- 路程是每一步移动距离之和，因此路程通常大于或等于位移距离。
- 平均速率等于总路程除以总时间。
- 浮点运算可能存在极小误差，测试中应使用 `pytest.approx` 或 `np.testing.assert_allclose`。