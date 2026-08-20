import numpy as np
import pytest

from robot_learning.pid_control import (
    PIDController,
    PIDGains,
    simulate_pid_point_mass,
)


def test_proportional_control_has_expected_sign() -> None:
    controller = PIDController(
        PIDGains(kp=2.0, ki=0.0, kd=0.0),
        output_limit=10.0,
        integral_limit=5.0,
    )

    assert controller.step(error=3.0, dt=0.1) == pytest.approx(6.0)
    assert controller.step(error=-1.0, dt=0.1) == pytest.approx(-2.0)


def test_control_output_is_limited() -> None:
    controller = PIDController(
        PIDGains(kp=100.0, ki=0.0, kd=0.0),
        output_limit=4.0,
        integral_limit=5.0,
    )

    assert controller.step(error=1.0, dt=0.1) == pytest.approx(4.0)


def test_integral_state_is_limited() -> None:
    controller = PIDController(
        PIDGains(kp=0.0, ki=1.0, kd=0.0),
        output_limit=10.0,
        integral_limit=0.5,
    )

    for _ in range(20):
        controller.step(error=1.0, dt=0.1)

    assert controller.integral == pytest.approx(0.5)


def test_derivative_responds_to_error_change() -> None:
    controller = PIDController(
        PIDGains(kp=0.0, ki=0.0, kd=2.0),
        output_limit=100.0,
        integral_limit=1.0,
    )

    assert controller.step(error=1.0, dt=0.1) == pytest.approx(0.0)
    assert controller.step(error=0.5, dt=0.1) == pytest.approx(-10.0)


def test_pid_point_mass_converges_near_target() -> None:
    simulation = simulate_pid_point_mass(
        initial_position=0.0,
        initial_velocity=0.0,
        target_position=5.0,
        mass=1.0,
        dt=0.01,
        steps=800,
        gains=PIDGains(kp=5.0, ki=0.1, kd=4.0),
        output_limit=12.0,
        integral_limit=2.0,
    )

    assert abs(simulation.positions[-1] - 5.0) < 0.05
    assert abs(simulation.velocities[-1]) < 0.05
    assert np.max(np.abs(simulation.controls)) <= 12.0


@pytest.mark.parametrize(
    ("mass", "dt", "steps"),
    [
        (0.0, 0.01, 10),
        (1.0, 0.0, 10),
        (1.0, 0.01, 0),
    ],
)
def test_simulation_rejects_invalid_parameters(
    mass: float,
    dt: float,
    steps: int,
) -> None:
    with pytest.raises(ValueError):
        simulate_pid_point_mass(
            initial_position=0.0,
            initial_velocity=0.0,
            target_position=1.0,
            mass=mass,
            dt=dt,
            steps=steps,
            gains=PIDGains(kp=1.0, ki=0.0, kd=0.0),
            output_limit=10.0,
            integral_limit=1.0,
        )
