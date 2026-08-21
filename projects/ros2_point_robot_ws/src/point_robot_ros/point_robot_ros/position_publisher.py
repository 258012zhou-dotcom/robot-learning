"""Publish simulated point robot positions."""

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_srvs.srv import Trigger


class PositionPublisher(Node):
    """Publish a point moving along the x axis."""

    def __init__(self) -> None:
        super().__init__("position_publisher")

        self._publisher = self.create_publisher(
            Point,
            "point_robot/position",
            10,
        )

        self._timer_period = 0.1
        self._velocity_x = 0.5
        self._position_x = 0.0

        self._timer = self.create_timer(
            self._timer_period,
            self._publish_position,
        )
        self._reset_service = self.create_service(
            Trigger,
            "point_robot/reset",
            self._reset_position,
        )

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
