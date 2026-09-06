"""Apply command limits, timeout stopping, and emergency stopping."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_srvs.srv import SetBool

from point_robot_ros.safety import SafetySupervisor


class SafetyNode(Node):
    """Publish only velocity commands allowed by the safety supervisor."""

    def __init__(self) -> None:
        super().__init__("safety_node")

        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("publish_period", 0.05)

        max_linear_speed = float(
            self.get_parameter("max_linear_speed").value
        )
        command_timeout = float(
            self.get_parameter("command_timeout").value
        )
        publish_period = float(
            self.get_parameter("publish_period").value
        )

        if not math.isfinite(publish_period) or publish_period <= 0.0:
            raise ValueError("publish_period must be finite and positive")

        self._supervisor = SafetySupervisor(
            max_command=max_linear_speed,
            command_timeout=command_timeout,
        )

        self._safe_publisher = self.create_publisher(
            Twist,
            "point_robot/cmd_vel_safe",
            10,
        )
        self._raw_subscription = self.create_subscription(
            Twist,
            "point_robot/cmd_vel_raw",
            self._handle_raw_command,
            10,
        )
        self._emergency_stop_service = self.create_service(
            SetBool,
            "point_robot/set_emergency_stop",
            self._handle_emergency_stop,
        )
        self._timer = self.create_timer(
            publish_period,
            self._publish_safe_command,
        )

        self.get_logger().info(
            "Safety node ready: max speed %.2f, timeout %.2f s"
            % (max_linear_speed, command_timeout)
        )

    def _handle_raw_command(self, message: Twist) -> None:
        """Validate and store the latest requested linear velocity."""
        try:
            accepted = self._supervisor.accept_command(
                command=float(message.linear.x),
                now=time.monotonic(),
            )
        except ValueError as error:
            # accept_command has cleared the previous velocity. Publish zero
            # immediately instead of waiting for the periodic timer.
            self._publish_safe_command()
            self.get_logger().warning(
                "Rejected invalid velocity command: %s" % error
            )
            return

        if not accepted:
            self.get_logger().debug(
                "Discarded velocity command while emergency stop is active"
            )

    def _handle_emergency_stop(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        """Engage or reset emergency stop through a ROS service."""
        if request.data:
            self._supervisor.engage_emergency_stop()
            self._publish_safe_command()
            response.message = "Emergency stop engaged"
        else:
            self._supervisor.reset_emergency_stop()
            response.message = (
                "Emergency stop reset; waiting for a fresh command"
            )

        response.success = True
        self.get_logger().warning(response.message)
        return response

    def _publish_safe_command(self) -> None:
        """Publish the velocity currently allowed by all safety checks."""
        message = Twist()
        message.linear.x = self._supervisor.safe_command(
            now=time.monotonic()
        )
        self._safe_publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    """Run the safety node until interrupted."""
    rclpy.init(args=args)
    node = SafetyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
