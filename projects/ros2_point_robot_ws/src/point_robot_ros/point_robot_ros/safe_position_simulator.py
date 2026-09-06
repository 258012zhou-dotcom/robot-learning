"""Integrate safe velocity commands into simulated one-dimensional motion."""

import math
import time

from geometry_msgs.msg import Point, Twist
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from point_robot_ros.safety import SafetySupervisor


class SafePositionSimulator(Node):
    """Simulate an actuator that accepts only the safety node's output."""

    def __init__(self) -> None:
        super().__init__("safe_position_simulator")
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("command_timeout", 0.5)
        self._position = float(self.get_parameter("initial_x").value)
        self._period = float(self.get_parameter("timer_period").value)
        if not math.isfinite(self._position):
            raise ValueError("initial_x must be finite")
        if not math.isfinite(self._period) or self._period <= 0.0:
            raise ValueError("timer_period must be positive and finite")

        # The simulated actuator has its own watchdog: if the safety node
        # disappears, a previously received nonzero command must also expire.
        self._commands = SafetySupervisor(
            max_command=float(self.get_parameter("max_linear_speed").value),
            command_timeout=float(self.get_parameter("command_timeout").value),
        )
        self._publisher = self.create_publisher(
            Point, "point_robot/position", 10,
        )
        self._subscription = self.create_subscription(
            Twist, "point_robot/cmd_vel_safe", self._accept_velocity, 10,
        )
        self._reset_service = self.create_service(
            Trigger, "point_robot/reset", self._reset_position,
        )
        self._timer = self.create_timer(self._period, self._step)

    def _accept_velocity(self, message: Twist) -> None:
        """Store safe velocity; invalid values discard any previous command."""
        try:
            self._commands.accept_command(
                float(message.linear.x), time.monotonic(),
            )
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _step(self) -> None:
        """Advance this teaching simulation by one fixed control interval."""
        velocity = self._commands.safe_command(time.monotonic())
        self._position += velocity * self._period
        message = Point()
        message.x = self._position
        self._publisher.publish(message)

    def _reset_position(self, _request, response):
        """Reset position and discard the currently stored velocity."""
        self._position = 0.0
        self._commands.discard_command()
        response.success = True
        response.message = "Simulated position reset to x=0.0"
        return response


def main(args: list[str] | None = None) -> None:
    """Run the simulated actuator without connecting to real hardware."""
    rclpy.init(args=args)
    node = SafePositionSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
