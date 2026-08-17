# 实验 003：最小传感器线程

## 目标

生产者线程生成确定性的二维位置样本；主线程通过队列接收它们。该实验只模拟线程通信，不连接真实传感器。

| 概念 | 作用 |
| --- | --- |
| `Thread` | 让生产者与主线程并发执行 |
| `Queue` | 安全传递样本；有限容量形成背压 |
| `Event` | 向生产者发送停止信号 |
| `None` sentinel | 明确通知消费者数据流结束 |
| `join()` | 等待线程退出，避免后台线程遗留 |

子线程中的异常不会自动传播到主线程：本实验用单独的错误队列传回异常。`Queue.get()` 也必须设置超时；`join(timeout=...)` 只能限制调用 `join()` 时的等待，不能保护发生在 `join()` 之前的 `get()` 阻塞。

配置在 `configs/003_sensor_thread.json`。运行：

```bash
conda activate robot_learning
./scripts/run_tests.sh
PYTHONPATH=src python experiments/003_sensor_thread/run.py
```

输出 `outputs/003_sensor_thread/results.json`，其中包含样本数、首末位置和线程停止状态。
