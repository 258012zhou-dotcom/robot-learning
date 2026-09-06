# 实验反思

## 历史结果与证据撤回

以下数字属于旧协议，保留用于追溯；single 的一条长轨迹与 vector 的多条短轨迹不等价，不能据此确认公平加速比。历史 outputs 保持不变，修复后尚未运行完整性能 benchmark。

- 轻负载下，单环境、Sync、Async 分别约为 26863、19950、11240 Transition/s。
- 轻负载 Async 只有单环境的 0.42 倍，进程通信与同步开销占主导。
- 较重负载下，三者分别约为 1466、1450、3670 Transition/s；Async 获得 2.50 倍加速。
- 较重负载的 Sync 为历史基准的 0.99 倍；这个数字本身不能证明并行机制。
- Async 初始化约 0.015 秒，明显高于单环境约 0.001 秒和 Sync 约 0.004 秒。
- 旧校验和相等只能说明总和相等，无法证明数值轨迹相同；撤回此前的轨迹一致性结论。

## 公平性修复

- 三种后端各运行 N 个环境，每个环境 seed、动作序列、预热步数和正式步数完全对应。single 现为串行池。
- 共用预热—原 seed reset—正式采样流程，初始化、预热、第二次 reset 与 rollout 分开计时。
- 轨迹 SHA-256 保留时间、环境及分量顺序，并编码数组形状；覆盖初始观测、动作、观测、奖励和结束标志。sum 仅作诊断。
- 关键回归断言先确认两数组 sum 相等，再确认其哈希不同；实际三后端运行要求哈希相同。二者分别检查“能发现差异”和“等价执行”。
- 哈希成本计入正式循环，结果包含采样与证据收集开销。不能用它宣称纯 MuJoCo 性能或学习性能提升。

## 结论

- 是否值得使用多进程，取决于单步环境计算量能否覆盖进程通信成本。
- 选择环境后端应以目标机器上的基准测试为依据，不能把“并行”直接等同于“更快”。
- 本实验提升的是单位时间采样量，即计算效率；没有证明算法减少了所需 Transition 数，因此没有提升样本效率。
- 初始化和稳定 rollout 是两类不同开销，短任务尤其不能忽略 worker 启动成本。

## 当前边界

- 没有包含神经网络推理、GPU 训练和经验回放缓冲区。
- 默认配置使用 4 个环境，没有搜索最佳 worker 数量。小规模验证只检查正确性，不给出新的性能排名。
- 哈希是数值输出的强证据，未比对 info、隐藏仿真状态；不同依赖版本或浮点实现可能导致精确哈希不同。
- 计时受 CPU 调频、后台程序和 WSL 负载影响，应关注数量级和趋势。
- 更高吞吐量只代表计算效率，不代表策略需要更少数据。

## 本次验证记录（2026-09-05）

- 单元与集成测试：执行 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_vector_environment_benchmark.py`，17 项通过。关键证据是等和异数组被区分、串行池逐环境输入及两次 reset 符合预期、三后端输出哈希一致、不同预热长度不改变正式轨迹、报告拒绝哈希不一致的等和结果。
- 实际运行：执行 `PYTHONPATH=src MPLBACKEND=Agg /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/023_vector_environment_benchmark/run.py --config /tmp/benchmark023-smoke-NU3IqE/config.json --output-dir /tmp/benchmark023-smoke-NU3IqE/results`，成功生成 CSV、JSON、图和日志。配置为 N=3、预热 2 步、每种负载 24 条 Transition、两次重复，frame_skip 分别为 5 和 100。
- 输出读回检查：CSV 共 12 行，每行 N=3、8 个批次、24 条 Transition；同负载六条记录哈希一致，JSON 全部复现检查通过。再次指向同一非空目录被拒绝，比较前后四个输出文件的 SHA-256，确认均未变化。
- 代码差异检查：`git diff --check -- src/robot_learning/vector_environment_benchmark.py tests/test_vector_environment_benchmark.py experiments/023_vector_environment_benchmark` 通过。这只验证空白格式，不是功能测试。
- 验证使用工作区现有的底层环境修复；未改动该环境文件。临时结果不属于 full benchmark，不替代历史性能记录。未新增依赖、未提交 Git。
