# 测试、调试与可复现性

pytest 收集 `tests/` 中的 `test_...` 函数并执行断言；参数化测试会按每组输入各运行一次。项目统一入口是 `./scripts/run_tests.sh`，它检查 Conda 环境并禁用无关 ROS pytest 插件。

调试时在 VS Code 设置断点，暂停后观察变量；`F10` 执行下一行而不进入函数。先读失败用例和报错，再决定修改位置。

可复现实验需要同一份代码、环境、配置和随机种子：

```text
environment.yml + JSON 配置 + seed + 日志/results.json → 可重复的输出
```

项目证据：实验 001 记录配置、种子和轨迹统计；实验 002 用 `np.random.default_rng(seed)`，固定 `seed=42` 得到相同 RMSE。只存图片而不存配置和结果，无法可靠复查实验。
