# NumPy 轨迹分析

机器人轨迹常表示为 `(T, D)`：`T` 是时间点数，`D` 是状态维度。实验 001 的二维轨迹是 `(101, 2)`。

- `np.diff(trajectory, axis=0)`：相邻位置的变化。
- `np.linalg.norm(vectors, axis=1)`：每一步向量长度。
- 位移是终点减起点；位移距离是直线距离；路程是每步距离之和，因此路程不小于位移距离。
- 平均速率 = 总路程 ÷ 总时间。

浮点数比较使用 `pytest.approx` 或 `np.testing.assert_allclose`。图像应回答问题：实验 001 用轨迹图检查空间路径、用速度图检查匀速假设；保存后 `plt.close(fig)`，输出可由代码和配置重新生成。
