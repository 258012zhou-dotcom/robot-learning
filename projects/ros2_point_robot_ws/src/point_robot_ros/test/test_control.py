"""Unit tests for basic feedback controllers."""

import pytest

from point_robot_ros.control import (
    PIDController,
    proportional_command,
    simulate_proportional_joint,
    simulate_pid_joint,
    simulate_second_order_joint,
)


def test_positive_error_produces_positive_command():
    command = proportional_command(
        target=1.0,
        current=0.25,
        kp=2.0,
        max_command=10.0,
    )

    assert command == pytest.approx(1.5)


def test_negative_error_produces_negative_command():
    command = proportional_command(
        target=-1.0,
        current=0.0,
        kp=2.0,
        max_command=10.0,
    )

    assert command == pytest.approx(-2.0)


def test_command_is_limited():
    command = proportional_command(
        target=10.0,
        current=0.0,
        kp=2.0,
        max_command=0.5,
    )

    assert command == pytest.approx(0.5)


def test_zero_error_produces_zero_command():
    command = proportional_command(
        target=1.0,
        current=1.0,
        kp=2.0,
        max_command=0.5,
    )

    assert command == pytest.approx(0.0)


def test_joint_moves_toward_positive_target():
    positions = simulate_proportional_joint(
        target=1.0,
        initial_position=0.0,
        kp=2.0,
        max_velocity=0.5,
        dt=0.1,
        steps=100,
    )

    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0, abs=1e-4)
    assert all(
        current <= following
        for current, following in zip(positions, positions[1:])
    )


def test_joint_moves_toward_negative_target():
    positions = simulate_proportional_joint(
        target=-1.0,
        initial_position=0.0,
        kp=2.0,
        max_velocity=0.5,
        dt=0.1,
        steps=100,
    )

    assert positions[-1] == pytest.approx(-1.0, abs=1e-4)


def test_joint_velocity_limit_is_respected():
    dt = 0.1
    max_velocity = 0.5

    positions = simulate_proportional_joint(
        target=10.0,
        initial_position=0.0,
        kp=2.0,
        max_velocity=max_velocity,
        dt=dt,
        steps=10,
    )

    position_changes = [
        following - current
        for current, following in zip(positions, positions[1:])
    ]

    max_position_change = max(
        abs(change) for change in position_changes
    )

    assert max_position_change == pytest.approx(
        max_velocity * dt
    )


def test_pid_proportional_term():
    controller = PIDController(
        kp=2.0,
        ki=0.0,
        kd=0.0,
        max_command=10.0,
        integral_limit=10.0,
    )

    command = controller.update(
        target=1.0,
        current=0.25,
        dt=0.1,
    )

    assert command == pytest.approx(1.5)


def test_pid_integral_term_accumulates_error():
    controller = PIDController(
        kp=0.0,
        ki=1.0,
        kd=0.0,
        max_command=10.0,
        integral_limit=10.0,
    )

    first_command = controller.update(
        target=1.0,
        current=0.0,
        dt=0.1,
    )
    second_command = controller.update(
        target=1.0,
        current=0.0,
        dt=0.1,
    )

    assert first_command == pytest.approx(0.1)
    assert second_command == pytest.approx(0.2)


def test_pid_derivative_term_responds_to_error_change():
    controller = PIDController(
        kp=0.0,
        ki=0.0,
        kd=1.0,
        max_command=10.0,
        integral_limit=10.0,
    )

    first_command = controller.update(
        target=0.0,
        current=0.0,
        dt=0.5,
    )
    second_command = controller.update(
        target=1.0,
        current=0.0,
        dt=0.5,
    )

    assert first_command == pytest.approx(0.0)
    assert second_command == pytest.approx(2.0)


def test_pid_integral_is_limited():
    controller = PIDController(
        kp=0.0,
        ki=1.0,
        kd=0.0,
        max_command=10.0,
        integral_limit=0.2,
    )

    command = 0.0
    for _ in range(10):
        command = controller.update(
            target=1.0,
            current=0.0,
            dt=0.1,
        )

    assert command == pytest.approx(0.2)


def test_pid_output_is_limited():
    controller = PIDController(
        kp=100.0,
        ki=0.0,
        kd=0.0,
        max_command=0.5,
        integral_limit=10.0,
    )

    command = controller.update(
        target=10.0,
        current=0.0,
        dt=0.1,
    )

    assert command == pytest.approx(0.5)


def test_pid_reset_clears_internal_state():
    controller = PIDController(
        kp=0.0,
        ki=1.0,
        kd=0.0,
        max_command=10.0,
        integral_limit=10.0,
    )

    controller.update(target=1.0, current=0.0, dt=0.1)
    controller.reset()

    command = controller.update(
        target=0.0,
        current=0.0,
        dt=0.1,
    )

    assert command == pytest.approx(0.0)


def test_integral_term_removes_steady_state_error():
    p_controller = PIDController(
        kp=2.0,
        ki=0.0,
        kd=0.0,
        max_command=1.0,
        integral_limit=1.0,
    )
    pi_controller = PIDController(
        kp=2.0,
        ki=1.0,
        kd=0.0,
        max_command=1.0,
        integral_limit=1.0,
    )

    p_positions = simulate_pid_joint(
        controller=p_controller,
        target=1.0,
        initial_position=0.0,
        velocity_disturbance=-0.1,
        dt=0.05,
        steps=400,
    )
    pi_positions = simulate_pid_joint(
        controller=pi_controller,
        target=1.0,
        initial_position=0.0,
        velocity_disturbance=-0.1,
        dt=0.05,
        steps=400,
    )

    p_error = abs(1.0 - p_positions[-1])
    pi_error = abs(1.0 - pi_positions[-1])

    assert p_positions[-1] == pytest.approx(0.95, abs=1e-3)
    assert pi_positions[-1] == pytest.approx(1.0, abs=1e-3)
    assert pi_error < p_error


def test_derivative_term_damps_inertial_joint():
    p_controller = PIDController(
        kp=4.0,
        ki=0.0,
        kd=0.0,
        max_command=10.0,
        integral_limit=1.0,
    )
    pd_controller = PIDController(
        kp=4.0,
        ki=0.0,
        kd=4.0,
        max_command=10.0,
        integral_limit=1.0,
    )

    p_positions = simulate_second_order_joint(
        controller=p_controller,
        target=1.0,
        initial_position=0.0,
        initial_velocity=0.0,
        inertia=1.0,
        damping=0.1,
        dt=0.01,
        steps=800,
    )
    pd_positions = simulate_second_order_joint(
        controller=pd_controller,
        target=1.0,
        initial_position=0.0,
        initial_velocity=0.0,
        inertia=1.0,
        damping=0.1,
        dt=0.01,
        steps=800,
    )

    assert max(p_positions) > 1.5
    assert max(pd_positions) < 1.05
    assert pd_positions[-1] == pytest.approx(1.0, abs=1e-2)
