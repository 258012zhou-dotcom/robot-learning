# 实验 023：Gymnasium 向量环境性能基准

## 目标

在相同逐环境工作负载下，比较普通环境串行池、`SyncVectorEnv` 和 `AsyncVectorEnv` 的采样吞吐量（throughput）、初始化开销及数值可复现性。

本实验不训练神经网络。它只隔离环境采样性能，避免把 GPU 训练时间、策略网络批处理和环境并行混成一个无法解释的数字。

## 三种后端

- `single`：保留历史配置名，现表示 4 个独立普通环境组成的串行池，按环境编号依次 step。
- `sync_vector`：4 个环境使用统一批量接口，但在同一进程依次执行。
- `async_vector`：4 个环境由独立 worker 进程执行，主进程发送 Action 并收集 Observation。

三种后端每轮都产生 4 条 Transition。总数为 T、环境数为 N 时，每个环境执行 T/N 步，要求整除。`vector_step_count` 对串行池也表示批次轮数；所有后端的环境数量和轨迹长度一致。

## 两种负载

- `light`：每条 Transition 包含 5 个 MuJoCo 物理步，测量轻环境中的进程通信开销。
- `heavier`：每条 Transition 包含 100 个物理步，观察环境计算加重后多进程是否更有价值。

不同负载改变了控制时间尺度，因此不能比较任务控制效果。这里只能在同一负载内比较后端速度。

## 公平性与复现性

- 每个 workload/backend 重复 3 次。
- 第 i 个环境使用 seed=`base_seed+i`；其第 t 步动作为 `float32(0.75*sin(0.017*t+0.37*i))`。三个后端和各次重复都相同。
- 每个环境预热相同步数，然后用原 seed 再次 reset，正式动作从 t=0 开始。遇到 terminated 或 truncated 直接报错，不把自动 reset 混入工作负载。
- 初始化包含全部 N 个环境的构造及首次 reset；预热、第二次 reset 单独记录；关闭不计入 rollout。
- rollout 包含动作生成、环境 step、批次组织、结束检查、sum 和 SHA-256 校验成本，属于该采样程序的端到端吞吐量，不是纯物理引擎速度。三者使用同一计时循环。
- SHA-256 覆盖第二次 reset 的初始观测，以及按时间步、环境编号、分量排列的动作、观测、奖励、terminated、truncated。每个数组编码维数、形状和 C 顺序数值，数值统一为小端 float64（float32 转换无损）。初始观测哈希在 rollout 计时前，逐步哈希在计时内。
- 所有后端、所有重复的轨迹哈希必须一致；sum checksum 仅保留作诊断，不能证明数组一致。哈希提供极强的一致性证据，但不等同于数学上无碰撞的逐元素证明，也不覆盖 info 或仿真内部隐藏状态。
- 时间本身受 CPU 频率和后台负载影响，不要求逐次完全相同。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_vector_environment_benchmark.py
PYTHONPATH=src python experiments/023_vector_environment_benchmark/run.py --output-dir /tmp/benchmark023-new-run
python -m json.tool /tmp/benchmark023-new-run/results.json
```

## 输出

- 输出目录必须新建或为空，拒绝覆盖历史结果。可用 `--config /path/to/small.json` 指定小规模配置；默认配置仍是完整 benchmark。
- `benchmark_repetitions.csv`：每次重复的初始化、预热、reset、rollout 时间、吞吐量、诊断 sum 和轨迹哈希。
- `results.json`：协议版本 2、seed/action 规则、计时与哈希范围、复现性检查和相对串行池加速比。`speedup_over_single` 字段保留，但基准已改为串行池。
- `vector_environment_benchmark.png`：吞吐量和初始化开销图。
- `run.log`：运行摘要。

## 历史实测结果（旧协议，不能用作公平加速比证据）

下表保留旧协议的历史记录：single 是一条长轨迹，vector 是多条短轨迹，工作负载不等价。表内加速比不能作为修复后协议的性能结论；旧输出未改动，本次未重跑 full benchmark。

| 负载 | 后端 | Transition/s | 相对单环境 |
| --- | --- | ---: | ---: |
| light | single | 26863 | 1.00x |
| light | sync_vector | 19950 | 0.74x |
| light | async_vector | 11240 | 0.42x |
| heavier | single | 1466 | 1.00x |
| heavier | sync_vector | 1450 | 0.99x |
| heavier | async_vector | 3670 | 2.50x |

旧 sum 一致不足以证明逐元素轨迹一致。历史数据提示可能存在通信开销与计算收益的权衡，但需用新协议重新测量后才能定量判断。

历史速度不能与新协议直接比较：串行环境数量、轨迹和校验成本均已改变。小规模 smoke 只验证运行链路，不用于判断谁更快。

## 结果解释原则

- `AsyncVectorEnv` 更快：环境计算足以抵消进程通信开销。
- `AsyncVectorEnv` 更慢：环境太轻、批量太小或进程开销占主导，不代表实现错误。
- `SyncVectorEnv` 更快：减少 Python 调用或批量组织可能带来收益，但它不是真正并行。
- 结果只代表当前机器、环境数量和 workload，不能直接推广到图像仿真或大型机器人。
- 这是仿真吞吐测试，不是学习性能测试：没有训练策略，不能证明奖励、成功率、收敛速度或样本效率改善，也没有真实机器人验证。
