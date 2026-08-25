"""Publish a minimal synthetic grayscale camera stream."""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def create_test_pattern(
    *,
    width: int,
    height: int,
    frame_index: int,
) -> bytes:
    """Create a grayscale gradient with a moving bright stripe."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    denominator = max(width - 1, 1)
    row = bytearray(
        (x * 180) // denominator
        for x in range(width)
    )

    stripe_start = (frame_index * 5) % width
    for offset in range(8):
        row[(stripe_start + offset) % width] = 255

    return bytes(row) * height


class SyntheticCameraPublisher(Node):
    """Publish synthetic Image and CameraInfo messages."""

    def __init__(self) -> None:
        super().__init__("synthetic_camera_publisher")

        self.declare_parameter("width", 320)
        self.declare_parameter("height", 240)
        self.declare_parameter("frame_rate", 5.0)
        self.declare_parameter("focal_length", 250.0)

        self._width = int(
            self.get_parameter("width").value
        )
        self._height = int(
            self.get_parameter("height").value
        )
        self._frame_rate = float(
            self.get_parameter("frame_rate").value
        )
        self._focal_length = float(
            self.get_parameter("focal_length").value
        )

        if self._width <= 0 or self._height <= 0:
            raise ValueError("image dimensions must be positive")
        if self._frame_rate <= 0.0:
            raise ValueError("frame_rate must be positive")
        if self._focal_length <= 0.0:
            raise ValueError("focal_length must be positive")

        self._image_publisher = self.create_publisher(
            Image,
            "camera/image_raw",
            qos_profile_sensor_data,
        )
        self._info_publisher = self.create_publisher(
            CameraInfo,
            "camera/camera_info",
            qos_profile_sensor_data,
        )

        self._frame_index = 0
        self._timer = self.create_timer(
            1.0 / self._frame_rate,
            self._publish_frame,
        )

    def _publish_frame(self) -> None:
        """Publish one image and its matching camera parameters."""
        stamp = self.get_clock().now().to_msg()
        frame_id = "camera_optical_frame"

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = frame_id
        image.height = self._height
        image.width = self._width
        image.encoding = "mono8"
        image.is_bigendian = 0
        image.step = self._width
        image.data = create_test_pattern(
            width=self._width,
            height=self._height,
            frame_index=self._frame_index,
        )

        camera_info = CameraInfo()
        camera_info.header.stamp = stamp
        camera_info.header.frame_id = frame_id
        camera_info.height = self._height
        camera_info.width = self._width
        camera_info.distortion_model = "plumb_bob"
        camera_info.d = [0.0] * 5

        cx = (self._width - 1) / 2.0
        cy = (self._height - 1) / 2.0
        focal_length = self._focal_length

        camera_info.k = [
            focal_length, 0.0, cx,
            0.0, focal_length, cy,
            0.0, 0.0, 1.0,
        ]
        camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        camera_info.p = [
            focal_length, 0.0, cx, 0.0,
            0.0, focal_length, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        self._image_publisher.publish(image)
        self._info_publisher.publish(camera_info)

        if self._frame_index % 10 == 0:
            self.get_logger().info(
                "Published frame %d: %dx%d mono8"
                % (
                    self._frame_index,
                    self._width,
                    self._height,
                )
            )

        self._frame_index += 1


def main(args: list[str] | None = None) -> None:
    """Run the synthetic camera publisher."""
    rclpy.init(args=args)
    node = SyntheticCameraPublisher()

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
