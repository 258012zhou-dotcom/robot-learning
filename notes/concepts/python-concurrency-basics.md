# Python 并发基础

## 快速复习

| 方式 | 内存关系 | 适合场景 |
| --- | --- | --- |
| 进程（process） | 独立 | CPU 密集任务、故障隔离 |
| 线程（thread） | 共享 | 阻塞 I/O、后台数据流 |
| `asyncio` | 单线程协作切换 | 大量等待型网络或 I/O 任务 |

线程共享内存，因此不要让多个线程随意改同一个列表。用安全的数据流替代共享状态：

```text
生产者线程 → Queue → 主线程
     Event ← 停止请求
```

`Queue` 传递数据，有限容量会产生背压；`None` 是结束标记；`Event` 可打断等待；`time.monotonic()` 适合间隔和超时；`join()` 确认线程退出。

## 项目中的用法与边界

实验 003 的生产者把 `SensorSample` 放入队列，异常写入错误队列，`finally` 发送 sentinel。主线程对 `Queue.get(timeout=...)` 和 `join(timeout=...)` 都设置上限：`join` 无法保护它之前已经发生的 `get` 阻塞。

`projects/python_concurrency_demo/` 显示：多进程的工作 PID 与主进程不同，结果经 `multiprocessing.Queue` 返回；`asyncio` 任务在 `await` 时切换，总等待时间接近最长任务，不会自动加速 CPU 计算。
