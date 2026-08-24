"""Publish simulated cumulative wheel encoder counts."""

import rclpy
from point_robot_interfaces.msg import WheelTicks
from rclpy.node import Node


class WheelEncoderPublisher(Node):
    """Publish deterministic left and right wheel tick counts."""

    def __init__(self) -> None:
        super().__init__("wheel_encoder_publisher")

        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("left_ticks_per_cycle", 10)
        self.declare_parameter("right_ticks_per_cycle", 10)

        self._timer_period = float(
            self.get_parameter("timer_period").value
        )
        self._left_ticks_per_cycle = int(
            self.get_parameter("left_ticks_per_cycle").value
        )
        self._right_ticks_per_cycle = int(
            self.get_parameter("right_ticks_per_cycle").value
        )

        if self._timer_period <= 0.0:
            raise ValueError("timer_period must be positive")

        self._left_count = 0
        self._right_count = 0

        self._publisher = self.create_publisher(
            WheelTicks,
            "wheel_ticks",
            10,
        )
        self._timer = self.create_timer(
            self._timer_period,
            self._publish_ticks,
        )

    def _publish_ticks(self) -> None:
        """Advance and publish cumulative encoder counts."""
        self._left_count += self._left_ticks_per_cycle
        self._right_count += self._right_ticks_per_cycle

        message = WheelTicks()
        message.stamp = self.get_clock().now().to_msg()
        message.left_count = self._left_count
        message.right_count = self._right_count

        self._publisher.publish(message)

        self.get_logger().info(
            "Publishing ticks: left=%d right=%d"
            % (message.left_count, message.right_count)
        )


def main(args: list[str] | None = None) -> None:
    """Run the simulated wheel encoder publisher."""
    rclpy.init(args=args)
    node = WheelEncoderPublisher()

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
