"""Request a point robot position reset."""

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Trigger


class ResetClient(Node):
    """Call the point robot reset service."""

    def __init__(self) -> None:
        super().__init__("reset_client")
        self._client = self.create_client(
            Trigger,
            "point_robot/reset",
        )

    def send_request(self) -> Future:
        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for reset service...")

        request = Trigger.Request()
        return self._client.call_async(request)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ResetClient()

    try:
        future = node.send_request()
        rclpy.spin_until_future_complete(node, future)

        response = future.result()
        node.get_logger().info(
            "Reset response: success=%s message=%s"
            % (response.success, response.message)
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
