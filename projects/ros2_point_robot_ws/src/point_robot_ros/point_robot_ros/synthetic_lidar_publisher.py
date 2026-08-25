"""Publish a minimal synthetic 2D laser scan."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def create_test_ranges(
    *,
    angle_min: float,
    angle_increment: float,
    beam_count: int,
) -> list[float]:
    """Create deterministic ranges for several virtual obstacles."""
    if beam_count < 2:
        raise ValueError("beam_count must be at least two")
    if angle_increment <= 0.0:
        raise ValueError("angle_increment must be positive")

    ranges: list[float] = []

    for index in range(beam_count):
        angle = angle_min + index * angle_increment

        if abs(angle) <= math.radians(10.0):
            distance = 1.5
        elif (
            math.radians(35.0)
            <= angle
            <= math.radians(50.0)
        ):
            distance = 2.0
        else:
            distance = 4.0

        ranges.append(distance)

    return ranges


class SyntheticLidarPublisher(Node):
    """Publish deterministic LaserScan messages."""

    def __init__(self) -> None:
        super().__init__("synthetic_lidar_publisher")

        self.declare_parameter("scan_rate", 5.0)
        self.declare_parameter("beam_count", 181)
        self.declare_parameter("range_min", 0.1)
        self.declare_parameter("range_max", 10.0)

        self._scan_rate = float(
            self.get_parameter("scan_rate").value
        )
        self._beam_count = int(
            self.get_parameter("beam_count").value
        )
        self._range_min = float(
            self.get_parameter("range_min").value
        )
        self._range_max = float(
            self.get_parameter("range_max").value
        )

        if self._scan_rate <= 0.0:
            raise ValueError("scan_rate must be positive")
        if self._beam_count < 2:
            raise ValueError("beam_count must be at least two")
        if self._range_min <= 0.0:
            raise ValueError("range_min must be positive")
        if self._range_max <= self._range_min:
            raise ValueError(
                "range_max must be greater than range_min"
            )

        self._angle_min = -math.pi / 2.0
        self._angle_max = math.pi / 2.0
        self._angle_increment = (
            self._angle_max - self._angle_min
        ) / (self._beam_count - 1)

        self._scan_time = 1.0 / self._scan_rate
        self._scan_index = 0

        self._publisher = self.create_publisher(
            LaserScan,
            "scan",
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(
            self._scan_time,
            self._publish_scan,
        )

    def _publish_scan(self) -> None:
        """Publish one synthetic laser scan."""
        message = LaserScan()

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        message.header.frame_id = "laser_frame"

        message.angle_min = self._angle_min
        message.angle_max = self._angle_max
        message.angle_increment = self._angle_increment

        message.scan_time = self._scan_time
        message.time_increment = (
            self._scan_time / self._beam_count
        )

        message.range_min = self._range_min
        message.range_max = self._range_max

        message.ranges = create_test_ranges(
            angle_min=self._angle_min,
            angle_increment=self._angle_increment,
            beam_count=self._beam_count,
        )
        message.intensities = []

        self._publisher.publish(message)

        if self._scan_index % 10 == 0:
            self.get_logger().info(
                "Published scan %d with %d beams"
                % (
                    self._scan_index,
                    self._beam_count,
                )
            )

        self._scan_index += 1


def main(args: list[str] | None = None) -> None:
    """Run the synthetic laser scanner."""
    rclpy.init(args=args)
    node = SyntheticLidarPublisher()

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
