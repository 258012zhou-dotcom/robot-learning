"""Broadcast the point robot position as a dynamic TF transform."""

import rclpy
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class PositionTfBroadcaster(Node):
    """Convert position messages into world-to-base_link transforms."""

    def __init__(self) -> None:
        super().__init__("position_tf_broadcaster")

        self._tf_broadcaster = TransformBroadcaster(self)

        self._subscription = self.create_subscription(
            Point,
            "point_robot/position",
            self._broadcast_transform,
            10,
        )

    def _broadcast_transform(self, message: Point) -> None:
        """Broadcast the latest robot position as a transform."""
        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = "base_link"

        transform.transform.translation.x = message.x
        transform.transform.translation.y = message.y
        transform.transform.translation.z = message.z

        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        self._tf_broadcaster.sendTransform(transform)


def main(args: list[str] | None = None) -> None:
    """Run the dynamic TF broadcaster node."""
    rclpy.init(args=args)
    node = PositionTfBroadcaster()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
