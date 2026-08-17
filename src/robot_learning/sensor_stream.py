"""用于线程通信的最小传感器样本流。"""

from dataclasses import dataclass
from numbers import Real
from queue import Queue
from threading import Event
import time


@dataclass(frozen=True)
class SensorSample:
    """一条不可变的二维位置传感器样本。"""

    sequence_id: int
    timestamp: float
    position: tuple[float, float]


def sensor_producer(
    output_queue: Queue[SensorSample | None],
    stop_event: Event,
    sample_count: int,
    interval: float,
    error_queue: Queue[Exception] | None = None,
) -> None:
    """生成确定性样本，并以 None 通知消费者流已结束。"""
    try:
        if type(sample_count) is not int or sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if (
            not isinstance(interval, Real)
            or isinstance(interval, bool)
            or interval < 0
        ):
            raise ValueError("interval must be a non-negative number")

        for sequence_id in range(sample_count):
            if stop_event.is_set():
                break

            sample = SensorSample(
                sequence_id=sequence_id,
                timestamp=time.monotonic(),
                position=(float(sequence_id), 0.5 * sequence_id),
            )
            output_queue.put(sample)

            if stop_event.wait(interval):
                break
    except Exception as error:
        if error_queue is None:
            raise
        error_queue.put(error)
    finally:
        output_queue.put(None)
