"""Unit tests for scalar Kalman state estimation."""

import pytest

from point_robot_ros.state_estimation import (
    ScalarKalmanFilter,
)


def test_prediction_updates_position_and_uncertainty():
    estimator = ScalarKalmanFilter(
        estimate=0.0,
        variance=1.0,
        process_variance=0.2,
    )

    estimator.predict(
        velocity=1.0,
        dt=2.0,
    )

    assert estimator.estimate == pytest.approx(2.0)
    assert estimator.variance == pytest.approx(1.2)


def test_update_matches_numeric_example():
    estimator = ScalarKalmanFilter(
        estimate=0.0,
        variance=1.0,
        process_variance=0.2,
    )

    estimator.predict(
        velocity=1.0,
        dt=1.0,
    )
    estimator.update(
        measurement=1.8,
        measurement_variance=0.3,
    )

    assert estimator.last_innovation == pytest.approx(0.8)
    assert estimator.last_gain == pytest.approx(0.8)
    assert estimator.estimate == pytest.approx(1.64)
    assert estimator.variance == pytest.approx(0.24)


def test_noisy_measurement_receives_lower_gain():
    accurate_sensor = ScalarKalmanFilter(
        estimate=0.0,
        variance=1.0,
        process_variance=0.0,
    )
    noisy_sensor = ScalarKalmanFilter(
        estimate=0.0,
        variance=1.0,
        process_variance=0.0,
    )

    accurate_sensor.update(
        measurement=10.0,
        measurement_variance=0.1,
    )
    noisy_sensor.update(
        measurement=10.0,
        measurement_variance=10.0,
    )

    assert accurate_sensor.last_gain > noisy_sensor.last_gain
    assert accurate_sensor.estimate > noisy_sensor.estimate


def test_repeated_measurements_reduce_uncertainty():
    estimator = ScalarKalmanFilter(
        estimate=0.0,
        variance=4.0,
        process_variance=0.0,
    )

    initial_variance = estimator.variance

    estimator.update(
        measurement=1.0,
        measurement_variance=1.0,
    )
    first_variance = estimator.variance

    estimator.update(
        measurement=1.0,
        measurement_variance=1.0,
    )
    second_variance = estimator.variance

    assert second_variance < first_variance
    assert first_variance < initial_variance


def test_filter_rejects_invalid_uncertainty():
    with pytest.raises(ValueError):
        ScalarKalmanFilter(
            estimate=0.0,
            variance=-1.0,
            process_variance=0.1,
        )

    estimator = ScalarKalmanFilter(
        estimate=0.0,
        variance=1.0,
        process_variance=0.1,
    )

    with pytest.raises(ValueError):
        estimator.update(
            measurement=1.0,
            measurement_variance=0.0,
        )
