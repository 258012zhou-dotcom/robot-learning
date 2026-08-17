"""运行最小传感器生产者线程实验。"""

import json
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

from robot_learning.sensor_stream import SensorSample, sensor_producer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "003_sensor_thread.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "003_sensor_thread"


def load_config() -> dict[str, Any]:
    """读取实验 JSON 配置。"""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """启动生产者线程，并在主线程中接收全部样本。"""
    config = load_config()
    output_queue: Queue[SensorSample | None] = Queue(
        maxsize=int(config["queue_capacity"])
    )
    stop_event = Event()
    producer_errors: Queue[Exception] = Queue()
    producer = Thread(
        target=sensor_producer,
        args=(
            output_queue,
            stop_event,
            int(config["sample_count"]),
            float(config["interval"]),
            producer_errors,
        ),
        name="sensor-producer",
        daemon=False,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("读取配置：%s", CONFIG_PATH)
    logger.info("启动线程：%s", producer.name)
    producer.start()

    received_samples: list[SensorSample] = []
    while True:
        try:
            item = output_queue.get(timeout=2.0)
        except Empty as error:
            stop_event.set()
            producer.join(timeout=2.0)
            if producer.is_alive():
                raise RuntimeError(
                    "sensor producer did not stop after queue timeout"
                ) from error
            raise RuntimeError(
                "timed out waiting for sensor producer output"
            ) from error

        if item is None:
            logger.info("收到结束标记")
            break
        received_samples.append(item)
        logger.info("收到样本 %s：%s", item.sequence_id, item.position)

    producer.join(timeout=2.0)
    if producer.is_alive():
        stop_event.set()
        producer.join(timeout=2.0)
        raise RuntimeError("sensor producer did not stop after join timeout")
    if not producer_errors.empty():
        original_error = producer_errors.get_nowait()
        raise RuntimeError("sensor producer failed") from original_error

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment_name": config["experiment_name"],
        "sample_count": len(received_samples),
        "first_position": (
            list(received_samples[0].position) if received_samples else None
        ),
        "last_position": (
            list(received_samples[-1].position) if received_samples else None
        ),
        "thread_stopped_normally": not producer.is_alive(),
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("实际接收样本数：%s", results["sample_count"])
    logger.info("线程正常停止：%s", results["thread_stopped_normally"])


if __name__ == "__main__":
    main()
