"""Unit tests for the command safety supervisor."""

import pytest

from point_robot_ros.safety import SafetySupervisor


def test_recent_command_passes_through() -> None:
    supervisor = SafetySupervisor(
        max_command=1.0,
        command_timeout=0.5,
    )

    accepted = supervisor.accept_command(command=0.4, now=10.0)

    assert accepted is True
    assert supervisor.safe_command(now=10.2) == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("command", "expected"),
    [(2.0, 1.0), (-2.0, -1.0)],
)
def test_command_is_limited(command: float, expected: float) -> None:
    supervisor = SafetySupervisor(
        max_command=1.0,
        command_timeout=0.5,
    )

    supervisor.accept_command(command=command, now=10.0)

    assert supervisor.safe_command(now=10.1) == pytest.approx(expected)


def test_stale_command_causes_stop() -> None:
    supervisor = SafetySupervisor(
        max_command=1.0,
        command_timeout=0.5,
    )
    supervisor.accept_command(command=0.4, now=10.0)

    assert supervisor.safe_command(now=10.6) == pytest.approx(0.0)


def test_emergency_stop_overrides_command() -> None:
    supervisor = SafetySupervisor(
        max_command=1.0,
        command_timeout=0.5,
    )
    supervisor.accept_command(command=0.4, now=10.0)

    supervisor.engage_emergency_stop()

    assert supervisor.emergency_stop_active is True
    assert supervisor.safe_command(now=10.1) == pytest.approx(0.0)


def test_reset_requires_a_fresh_command() -> None:
    supervisor = SafetySupervisor(
        max_command=1.0,
        command_timeout=0.5,
    )
    supervisor.accept_command(command=0.4, now=10.0)
    supervisor.engage_emergency_stop()

    accepted_during_stop = supervisor.accept_command(
        command=0.8,
        now=10.1,
    )
    supervisor.reset_emergency_stop()

    assert accepted_during_stop is False
    assert supervisor.emergency_stop_active is False
    assert supervisor.safe_command(now=10.2) == pytest.approx(0.0)

    assert supervisor.accept_command(command=0.2, now=10.3) is True
    assert supervisor.safe_command(now=10.4) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("max_command", "command_timeout"),
    [(0.0, 0.5), (1.0, 0.0), (float("inf"), 0.5)],
)
def test_invalid_configuration_is_rejected(
    max_command: float,
    command_timeout: float,
) -> None:
    with pytest.raises(ValueError):
        SafetySupervisor(
            max_command=max_command,
            command_timeout=command_timeout,
        )
