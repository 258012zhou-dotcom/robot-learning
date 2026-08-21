"""Send movement goals to the point robot Action Server."""

import rclpy
from point_robot_interfaces.action import MoveToPosition
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future


class MoveActionClient(Node):
    """Send a movement goal and display its progress."""

    def __init__(self) -> None:
        super().__init__("move_action_client")
        self._action_client = ActionClient(
            self,
            MoveToPosition,
            "point_robot/move_to_position",
        )

    def send_goal(
        self,
        target_x: float,
        max_speed: float,
    ) -> Future:
        while not self._action_client.wait_for_server(
            timeout_sec=1.0,
        ):
            self.get_logger().info(
                "Waiting for move Action Server..."
            )

        goal = MoveToPosition.Goal()
        goal.target_x = target_x
        goal.max_speed = max_speed

        return self._action_client.send_goal_async(
            goal,
            feedback_callback=self._receive_feedback,
        )

    def _receive_feedback(self, feedback_message) -> None:
        feedback = feedback_message.feedback
        self.get_logger().info(
            "Feedback: current_x=%.2f remaining=%.2f"
            % (
                feedback.current_x,
                feedback.remaining_distance,
            )
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MoveActionClient()

    try:
        goal_future = node.send_goal(
            target_x=1.0,
            max_speed=0.2,
        )
        rclpy.spin_until_future_complete(node, goal_future)

        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            node.get_logger().info("Goal rejected")
            return

        node.get_logger().info("Goal accepted")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)

        wrapped_result = result_future.result()
        result = wrapped_result.result

        node.get_logger().info(
            "Result: success=%s final_x=%.2f message=%s"
            % (
                result.success,
                result.final_x,
                result.message,
            )
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
