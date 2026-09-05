# 实验 023：Gymnasium 向量环境性能基准

## 目标

在固定总 Transition 数的条件下，比较普通单环境、`SyncVectorEnv` 和 `AsyncVectorEnv` 的稳定采样吞吐量、初始化开销及数值可复现性。

本实验不训练神经网络。它只隔离环境采样性能，避免把 GPU 训练时间、策略网络批处理和环境并行混成一个无法解释的数字。

## 三种后端

- `single`：一个普通环境逐步运行。
- `sync_vector`：4 个环境使用统一批量接口，但在同一进程依次执行。
- `async_vector`：4 个环境由独立 worker 进程执行，主进程发送 Action 并收集 Observation。

向量环境每次调用产生 4 条 Transition，所以它的 vector step 数是单环境的四分之一。比较时固定总 Transition 数，而不是固定 API 调用次数。

## 两种负载

- `light`：每条 Transition 包含 5 个 MuJoCo 物理步，测量轻环境中的进程通信开销。
- `heavier`：每条 Transition 包含 100 个物理步，观察环境计算加重后多进程是否更有价值。

不同负载改变了控制时间尺度，因此不能比较任务控制效果。这里只能在同一负载内比较后端速度。

## 公平性与复现性

- 每个 workload/backend 重复 3 次。
- 每次使用相同环境 seed 和确定性 Action 序列。
- 正式计时前预热，再恢复相同初始状态。
- 初始化时间与稳定 rollout 时间分开。
- Sync 和 Async 必须产生相同数值校验值。
- 时间本身受 CPU 频率和后台负载影响，不要求逐次完全相同。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_vector_environment_benchmark.py
PYTHONPATH=src python experiments/023_vector_environment_benchmark/run.py
python -m json.tool outputs/023_vector_environment_benchmark/results.json
```

## 输出

- `benchmark_repetitions.csv`：每次重复的原始时间、吞吐量和校验值。
- `results.json`：复现性检查、吞吐量汇总与相对单环境加速比。
- `vector_environment_benchmark.png`：吞吐量和初始化开销图。
- `run.log`：运行摘要。

## 本机实测结果

下表使用 3 次重复的吞吐量中位数；加速比均以同一负载下的单环境为基准。

| 负载 | 后端 | Transition/s | 相对单环境 |
| --- | --- | ---: | ---: |
| light | single | 26863 | 1.00x |
| light | sync_vector | 19950 | 0.74x |
| light | async_vector | 11240 | 0.42x |
| heavier | single | 1466 | 1.00x |
| heavier | sync_vector | 1450 | 0.99x |
| heavier | async_vector | 3670 | 2.50x |

轻负载中，进程间通信成本高于并行收益；计算量增大后，4 个异步 worker 才产生明确加速。所有重复运行的数值校验和一致，Sync 与 Async 的校验和也一致。

这些速度是当前机器上的一次实测，不应当作为其他机器的固定性能指标。重新运行时应关注结论趋势，而不是要求吞吐量数字完全相同。

## 结果解释原则

- `AsyncVectorEnv` 更快：环境计算足以抵消进程通信开销。
- `AsyncVectorEnv` 更慢：环境太轻、批量太小或进程开销占主导，不代表实现错误。
- `SyncVectorEnv` 更快：减少 Python 调用或批量组织可能带来收益，但它不是真正并行。
- 结果只代表当前机器、环境数量和 workload，不能直接推广到图像仿真或大型机器人。
