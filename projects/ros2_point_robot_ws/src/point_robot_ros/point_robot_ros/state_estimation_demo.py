"""Compare measurement, prediction, and Kalman estimates."""

import math

from point_robot_ros.state_estimation import (
    ScalarKalmanFilter,
)


MEASUREMENT_ERRORS = [
    0.8,
    -0.4,
    0.6,
    -0.7,
    0.2,
    0.5,
    -0.3,
    0.4,
    -0.6,
    0.1,
]


def calculate_rmse(
    values: list[float],
    truth: list[float],
) -> float:
    """Calculate root mean squared error."""
    squared_errors = [
        (value - target) ** 2
        for value, target in zip(values, truth)
    ]
    return math.sqrt(
        sum(squared_errors) / len(squared_errors)
    )


def main() -> None:
    """Run a deterministic scalar state-estimation demo."""
    estimator = ScalarKalmanFilter(
        estimate=0.0,
        variance=1.0,
        process_variance=0.05,
    )

    true_positions: list[float] = []
    measurements: list[float] = []
    predictions: list[float] = []
    estimates: list[float] = []

    true_velocity = 1.0
    model_velocity = 0.9
    dt = 1.0
    predicted_position = 0.0

    print(
        "step | truth | measurement | prediction | "
        "estimate | gain"
    )

    for step, measurement_error in enumerate(
        MEASUREMENT_ERRORS,
        start=1,
    ):
        true_position = true_velocity * step * dt
        measurement = (
            true_position + measurement_error
        )

        predicted_position += model_velocity * dt

        estimator.predict(
            velocity=model_velocity,
            dt=dt,
        )
        estimator.update(
            measurement=measurement,
            measurement_variance=0.25,
        )

        true_positions.append(true_position)
        measurements.append(measurement)
        predictions.append(predicted_position)
        estimates.append(estimator.estimate)

        print(
            f"{step:>4} | "
            f"{true_position:>5.2f} | "
            f"{measurement:>11.2f} | "
            f"{predicted_position:>10.2f} | "
            f"{estimator.estimate:>8.2f} | "
            f"{estimator.last_gain:.3f}"
        )

    measurement_rmse = calculate_rmse(
        measurements,
        true_positions,
    )
    prediction_rmse = calculate_rmse(
        predictions,
        true_positions,
    )
    estimate_rmse = calculate_rmse(
        estimates,
        true_positions,
    )

    print()
    print(f"Measurement RMSE: {measurement_rmse:.4f}")
    print(f"Prediction RMSE:  {prediction_rmse:.4f}")
    print(f"Kalman RMSE:      {estimate_rmse:.4f}")


if __name__ == "__main__":
    main()
