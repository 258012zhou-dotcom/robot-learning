"""Move a simulated point robot toward a target position."""

import math
import time

import rclpy
from geometry_msgs.msg import Point
from point_robot_interfaces.action import MoveToPosition
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node


class MoveActionServer(Node):
    """Execute point robot movement goals."""

    def __init__(self) -> None:
        super().__init__("move_action_server")

        self._current_x = 0.0
        self._time_step = 0.1

        self._position_publisher = self.create_publisher(
            Point,
            "point_robot/position",
            10,
        )
        self._action_server = ActionServer(
            self,
            MoveToPosition,
            "point_robot/move_to_position",
            self._execute_goal,
            cancel_callback=self._reject_cancel,
        )

    def _reject_cancel(self, _goal_handle) -> CancelResponse:
        """Reject cancellation until concurrent execution is introduced."""
        return CancelResponse.REJECT

    def _execute_goal(
        self,
        goal_handle,
    ) -> MoveToPosition.Result:
        """Move toward the requested target and publish progress."""
        target_x = float(goal_handle.request.target_x)
        max_speed = float(goal_handle.request.max_speed)
        result = MoveToPosition.Result()

        if not math.isfinite(target_x):
            goal_handle.abort()
            result.success = False
            result.final_x = self._current_x
            result.message = "target_x must be finite"
            return result

        if not math.isfinite(max_speed) or max_speed <= 0.0:
            goal_handle.abort()
            result.success = False
            result.final_x = self._current_x
            result.message = "max_speed must be positive"
            return result

        self.get_logger().info(
            "Moving from %.2f to %.2f at max speed %.2f"
            % (self._current_x, target_x, max_speed)
        )

        while not math.isclose(
            self._current_x,
            target_x,
            abs_tol=1e-9,
        ):
            distance = target_x - self._current_x
            step = min(
                max_speed * self._time_step,
                abs(distance),
            )
            direction = 1.0 if distance > 0.0 else -1.0
            self._current_x += direction * step

            message = Point()
            message.x = self._current_x
            message.y = 0.0
            message.z = 0.0
            self._position_publisher.publish(message)

            feedback = MoveToPosition.Feedback()
            feedback.current_x = self._current_x
            feedback.remaining_distance = abs(
                target_x - self._current_x
            )
            goal_handle.publish_feedback(feedback)

            time.sleep(self._time_step)

        goal_handle.succeed()
        result.success = True
        result.final_x = self._current_x
        result.message = "Target reached"
        return result


def main(args: list[str] | None = None) -> None:
    """Run the move Action Server until interrupted."""
    rclpy.init(args=args)
    node = MoveActionServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
