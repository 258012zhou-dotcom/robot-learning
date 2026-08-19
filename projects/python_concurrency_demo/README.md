# Python Concurrency Demo

演示 Python 多进程和 `asyncio` 的基本运行边界。

## 运行

```bash
python projects/python_concurrency_demo/main.py
```

## 观察重点

- 多进程任务在独立进程中执行，工作进程的 PID 与主进程不同。
- `asyncio` 任务仍在同一进程中，通过等待点交替推进。
- 两个各自等待约 0.3 秒的异步任务并发执行时，总耗时应接近 0.3 秒，而不是二者相加。

该程序用于理解并发模型，不是性能基准；实际耗时会受机器负载影响。
