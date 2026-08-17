# 测试、调试与可复现性

## 核心概念

- **pytest** 会收集 `tests/` 中的 `test_...` 函数并执行断言。`@pytest.mark.parametrize` 会把一组测试数据展开为多次独立检查。
- **断点调试（breakpoint debugging）**：在 VS Code 代码行左侧设置断点，运行调试后暂停程序，检查变量和调用过程；`F10` 执行下一行但不进入函数。
- **环境隔离（environment isolation）**：项目使用 `robot_learning` Conda 环境，依赖记录在 `environment.yml`。测试脚本禁用未使用的 ROS pytest 插件，避免外部环境干扰。
- **配置、日志和随机种子**：JSON 配置记录参数；日志记录一次运行的关键事实；固定随机种子让随机实验可以重复。
- **可复现输出（reproducible output）**：在相同代码、配置、环境和种子下，应能重新得到相同的结果 JSON 和图像；生成物不代替源码和配置。

## 项目中的实际证据

- `./scripts/run_tests.sh` 会检查 Conda 环境，并运行 `python -m pytest -q`。
- 测试覆盖正常结果与非法输入：例如 `PositionBuffer` 拒绝错误容量和非二维位置，`PointRobotConfig` 拒绝错误字段和类型。
- 实验 001 将配置、随机种子、轨迹形状、最终位置和统计量写入日志与 `results.json`。
- 实验 002 使用 `np.random.default_rng(seed)`；在固定 `seed=42` 下，得到原始 RMSE `0.6224`、过滤后 RMSE `0.2543`。

## 常见错误

- 直接运行 `python tests/test_x.py`，而不使用 pytest：测试收集、项目路径和报告行为可能不正确。
- 测试失败后只修改代码、不先读报错和失败用例：容易修错问题。
- 在错误的 Conda 环境中运行：可能缺少依赖，或加载无关插件。
- 随机实验不记录种子：无法确认两次结果差异来自代码还是随机性。
- 只保存图片而没有配置或结果 JSON：无法复查输入条件。

## 复习问题

1. 一个参数化测试为何会让 pytest 的通过数量增加多次？
2. 调试时，什么情况下应在函数入口设置断点？
3. 要复现实验 002 的 RMSE，至少需要保留哪些信息？
4. 为什么测试脚本要禁用不使用的 ROS pytest 插件？
