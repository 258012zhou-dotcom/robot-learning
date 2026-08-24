"""Publish simulated point robot positions."""

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_srvs.srv import Trigger
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter


class PositionPublisher(Node):
    """Publish a point moving along the x axis."""

    def __init__(self) -> None:
        super().__init__("position_publisher")

        self._publisher = self.create_publisher(
            Point,
            "point_robot/position",
            10,
        )

        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("velocity_x", 0.5)
        self.declare_parameter("initial_x", 0.0)

        self._timer_period = float(
            self.get_parameter("timer_period").value
        )
        self._velocity_x = float(
            self.get_parameter("velocity_x").value
        )
        self._position_x = float(
            self.get_parameter("initial_x").value
        )

        if self._timer_period <= 0.0:
            raise ValueError("timer_period must be positive")

        self.add_on_set_parameters_callback(self._update_parameters)

        self._timer = self.create_timer(
            self._timer_period,
            self._publish_position,
        )
        self._reset_service = self.create_service(
            Trigger,
            "point_robot/reset",
            self._reset_position,
        )

    def _update_parameters(
        self,
        parameters: list[Parameter],
    ) -> SetParametersResult:
        """Validate and apply runtime parameter updates."""
        for parameter in parameters:
            if parameter.name in {"initial_x", "timer_period"}:
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} is startup-only",
                )

            if (
                parameter.name == "velocity_x"
                and parameter.type_ != Parameter.Type.DOUBLE
            ):
                return SetParametersResult(
                    successful=False,
                    reason="velocity_x must be a double",
                )

        for parameter in parameters:
            if parameter.name == "velocity_x":
                self._velocity_x = float(parameter.value)
                self.get_logger().info(
                    "Updated velocity_x to %.2f" % self._velocity_x
                )

        return SetParametersResult(successful=True)

    def _publish_position(self) -> None:
        message = Point()
        message.x = self._position_x
        message.y = 0.0
        message.z = 0.0

        self._publisher.publish(message)

        self.get_logger().info(
            "Publishing position: x=%.2f y=%.2f"
            % (message.x, message.y)
        )

        self._position_x += self._velocity_x * self._timer_period

    def _reset_position(
        self,
        _request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        self._position_x = 0.0

        response.success = True
        response.message = "Position reset to x=0.0"

        self.get_logger().info(response.message)
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PositionPublisher()

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
