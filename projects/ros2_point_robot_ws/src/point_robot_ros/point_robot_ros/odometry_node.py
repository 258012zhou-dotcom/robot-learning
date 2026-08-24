"""Estimate and publish differential-drive odometry."""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from point_robot_interfaces.msg import WheelTicks
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from point_robot_ros.odometry import (
    DifferentialDriveOdometry,
    normalize_angle,
)
from point_robot_ros.wheel_encoder import WheelEncoderModel


def message_time_seconds(message: WheelTicks) -> float:
    """Convert a ROS timestamp into floating-point seconds."""
    return (
        float(message.stamp.sec)
        + float(message.stamp.nanosec) * 1e-9
    )


class OdometryNode(Node):
    """Convert cumulative wheel ticks into robot odometry."""

    def __init__(self) -> None:
        super().__init__("odometry_node")

        self.declare_parameter("wheel_radius", 0.05)
        self.declare_parameter("ticks_per_revolution", 1000)
        self.declare_parameter("wheel_base", 0.30)

        wheel_radius = float(
            self.get_parameter("wheel_radius").value
        )
        ticks_per_revolution = int(
            self.get_parameter("ticks_per_revolution").value
        )
        wheel_base = float(
            self.get_parameter("wheel_base").value
        )

        left_encoder = WheelEncoderModel(
            wheel_radius=wheel_radius,
            ticks_per_revolution=ticks_per_revolution,
        )
        right_encoder = WheelEncoderModel(
            wheel_radius=wheel_radius,
            ticks_per_revolution=ticks_per_revolution,
        )

        self._odometry = DifferentialDriveOdometry(
            left_encoder=left_encoder,
            right_encoder=right_encoder,
            wheel_base=wheel_base,
        )

        self._previous_time: float | None = None

        self._publisher = self.create_publisher(
            Odometry,
            "odom",
            10,
        )
        self._subscription = self.create_subscription(
            WheelTicks,
            "wheel_ticks",
            self._handle_ticks,
            10,
        )
        self._tf_broadcaster = TransformBroadcaster(self)

    def _handle_ticks(self, message: WheelTicks) -> None:
        """Update and publish the latest odometry estimate."""
        previous_pose = self._odometry.pose

        pose = self._odometry.update(
            left_count=message.left_count,
            right_count=message.right_count,
        )

        current_time = message_time_seconds(message)
        linear_velocity = 0.0
        angular_velocity = 0.0

        if self._previous_time is not None:
            dt = current_time - self._previous_time

            if dt > 0.0:
                heading_change = normalize_angle(
                    pose.heading - previous_pose.heading
                )
                middle_heading = normalize_angle(
                    previous_pose.heading
                    + heading_change / 2.0
                )

                world_dx = pose.x - previous_pose.x
                world_dy = pose.y - previous_pose.y

                forward_distance = (
                    world_dx * math.cos(middle_heading)
                    + world_dy * math.sin(middle_heading)
                )

                linear_velocity = forward_distance / dt
                angular_velocity = heading_change / dt

        self._previous_time = current_time

        odometry_message = Odometry()
        odometry_message.header.stamp = message.stamp
        odometry_message.header.frame_id = "odom"
        odometry_message.child_frame_id = "base_link"

        odometry_message.pose.pose.position.x = pose.x
        odometry_message.pose.pose.position.y = pose.y

        odometry_message.pose.pose.orientation.z = math.sin(
            pose.heading / 2.0
        )
        odometry_message.pose.pose.orientation.w = math.cos(
            pose.heading / 2.0
        )

        odometry_message.twist.twist.linear.x = linear_velocity
        odometry_message.twist.twist.angular.z = angular_velocity

        self._publisher.publish(odometry_message)
        self._broadcast_transform(
            message=odometry_message,
        )

        self.get_logger().info(
            "Odometry: x=%.3f y=%.3f heading=%.3f"
            % (pose.x, pose.y, pose.heading)
        )

    def _broadcast_transform(
        self,
        *,
        message: Odometry,
    ) -> None:
        """Broadcast odom-to-base_link from the odometry estimate."""
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = message.child_frame_id

        transform.transform.translation.x = (
            message.pose.pose.position.x
        )
        transform.transform.translation.y = (
            message.pose.pose.position.y
        )
        transform.transform.rotation = (
            message.pose.pose.orientation
        )

        self._tf_broadcaster.sendTransform(transform)


def main(args: list[str] | None = None) -> None:
    """Run the odometry node."""
    rclpy.init(args=args)
    node = OdometryNode()

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
