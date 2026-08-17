import asyncio
from multiprocessing import Process, Queue
import os
import time


def cpu_worker(output_queue: Queue) -> None:
    """在独立进程中执行一段 CPU 计算。"""
    result = sum(number * number for number in range(1_000_000))
    output_queue.put(
        {
            "worker_pid": os.getpid(),
            "result": result,
        }
    )


def run_process_demo() -> None:
    """启动子进程，并通过多进程队列接收结果。"""
    output_queue = Queue()
    worker = Process(
        target=cpu_worker,
        args=(output_queue,),
        name="cpu-worker",
    )

    print(f"main process pid: {os.getpid()}")

    worker.start()
    message = output_queue.get(timeout=5.0)
    worker.join(timeout=5.0)

    if worker.is_alive():
        worker.terminate()
        worker.join()
        raise RuntimeError("worker process did not stop")

    print(f"worker process pid: {message['worker_pid']}")
    print(f"process result: {message['result']}")


async def simulated_io(name: str, delay: float) -> str:
    """模拟一个需要等待的 I/O 任务。"""
    print(f"{name} started, pid: {os.getpid()}")
    await asyncio.sleep(delay)
    print(f"{name} finished")
    return name


async def run_async_demo() -> None:
    """让两个等待型任务并发执行。"""
    start_time = time.monotonic()

    completed_tasks = await asyncio.gather(
        simulated_io("camera", 0.2),
        simulated_io("network", 0.3),
    )

    elapsed_time = time.monotonic() - start_time

    print(f"async results: {completed_tasks}")
    print(f"async elapsed time: {elapsed_time:.3f} seconds")


def main() -> None:
    print("=== Process demo ===")
    run_process_demo()

    print("\n=== Async demo ===")
    asyncio.run(run_async_demo())


if __name__ == "__main__":
    main()