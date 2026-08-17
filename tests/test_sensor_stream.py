from queue import Queue
from threading import Event

import pytest

from robot_learning.sensor_stream import SensorSample, sensor_producer


def test_producer_generates_samples_in_order_then_sentinel():
    output_queue: Queue[SensorSample | None] = Queue()

    sensor_producer(output_queue, Event(), sample_count=3, interval=0)

    samples = [output_queue.get_nowait() for _ in range(3)]
    assert [sample.sequence_id for sample in samples] == [0, 1, 2]
    assert [sample.position for sample in samples] == [
        (0.0, 0.0),
        (1.0, 0.5),
        (2.0, 1.0),
    ]
    assert output_queue.get_nowait() is None


def test_producer_sends_sentinel_after_zero_samples():
    output_queue: Queue[SensorSample | None] = Queue()

    sensor_producer(output_queue, Event(), sample_count=0, interval=0)

    assert output_queue.get_nowait() is None


def test_previously_stopped_producer_sends_no_regular_samples():
    output_queue: Queue[SensorSample | None] = Queue()
    stop_event = Event()
    stop_event.set()

    sensor_producer(output_queue, stop_event, sample_count=3, interval=0)

    assert output_queue.get_nowait() is None
    assert output_queue.empty()


def test_invalid_parameter_still_sends_sentinel():
    output_queue: Queue[SensorSample | None] = Queue()

    with pytest.raises(ValueError, match="sample_count"):
        sensor_producer(output_queue, Event(), sample_count=-1, interval=0)

    assert output_queue.get_nowait() is None


def test_producer_reports_error_to_error_queue_when_provided():
    output_queue: Queue[SensorSample | None] = Queue()
    producer_errors: Queue[Exception] = Queue()

    sensor_producer(
        output_queue,
        Event(),
        sample_count=-1,
        interval=0,
        error_queue=producer_errors,
    )

    assert isinstance(producer_errors.get_nowait(), ValueError)
    assert output_queue.get_nowait() is None


@pytest.mark.parametrize("sample_count", [-1, True, 1.5])
def test_producer_rejects_invalid_sample_count(sample_count: object):
    with pytest.raises(ValueError, match="sample_count"):
        sensor_producer(Queue(), Event(), sample_count=sample_count, interval=0)


@pytest.mark.parametrize("interval", [-0.1, True, "0.1"])
def test_producer_rejects_invalid_interval(interval: object):
    with pytest.raises(ValueError, match="interval"):
        sensor_producer(Queue(), Event(), sample_count=0, interval=interval)
