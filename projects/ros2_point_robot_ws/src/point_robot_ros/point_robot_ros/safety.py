"""Safety checks applied after control commands are computed."""

from dataclasses import dataclass, field
import math


@dataclass
class SafetySupervisor:
    """Limit commands and stop on timeout or emergency-stop activation."""

    max_command: float
    command_timeout: float

    _command: float = field(default=0.0, init=False)
    _last_command_time: float | None = field(default=None, init=False)
    _emergency_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_command) or self.max_command <= 0.0:
            raise ValueError("max_command must be finite and positive")
        if (
            not math.isfinite(self.command_timeout)
            or self.command_timeout <= 0.0
        ):
            raise ValueError("command_timeout must be finite and positive")

    @property
    def emergency_stop_active(self) -> bool:
        """Return whether the emergency stop is currently active."""
        return self._emergency_stop

    def accept_command(self, command: float, now: float) -> bool:
        """Store a finite, limited command unless emergency stop is active."""
        try:
            self._validate_finite(command=command, now=now)
        except ValueError:
            # Rejecting a new invalid command must also discard old motion.
            self.discard_command()
            raise

        if self._emergency_stop:
            return False

        self._command = max(
            -self.max_command,
            min(self.max_command, command),
        )
        self._last_command_time = now
        return True

    def safe_command(self, now: float) -> float:
        """Return the command allowed by the current safety state."""
        if not math.isfinite(now):
            self.discard_command()
            raise ValueError("now must be finite")

        if self._emergency_stop or self._last_command_time is None:
            return 0.0

        elapsed = now - self._last_command_time
        if elapsed < 0.0:
            self.discard_command()
            raise ValueError("now cannot move backwards")
        if elapsed > self.command_timeout:
            self.discard_command()
            return 0.0

        return self._command

    def discard_command(self) -> None:
        """Clear motion until a fresh valid command arrives, keeping the latch."""
        self._command = 0.0
        self._last_command_time = None

    def engage_emergency_stop(self) -> None:
        """Stop immediately and discard the previously accepted command."""
        self._emergency_stop = True
        self.discard_command()

    def reset_emergency_stop(self) -> None:
        """Release emergency stop without restoring an old command."""
        self._emergency_stop = False

    @staticmethod
    def _validate_finite(*, command: float, now: float) -> None:
        if not math.isfinite(command):
            raise ValueError("command must be finite")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
