"""Subscribe to simulated point robot positions."""

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node


class PositionSubscriber(Node):
    """Receive and display point robot positions."""

    def __init__(self) -> None:
        super().__init__("position_subscriber")

        self._subscription = self.create_subscription(
            Point,
            "point_robot/position",
            self._receive_position,
            10,
        )

    def _receive_position(self, message: Point) -> None:
        self.get_logger().info(
            "Received position: x=%.2f y=%.2f"
            % (message.x, message.y)
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PositionSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
