"""Move a simulated point robot toward a target position."""

import math
from threading import Lock
import time

import rclpy
from geometry_msgs.msg import Point
from point_robot_interfaces.action import MoveToPosition
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
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
        self._goal_lock = Lock()
        self._goal_active = False
        self._callback_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            MoveToPosition,
            "point_robot/move_to_position",
            self._execute_goal,
            goal_callback=self._handle_goal,
            cancel_callback=self._accept_cancel,
            callback_group=self._callback_group,
        )

    def _handle_goal(
        self,
        goal_request: MoveToPosition.Goal,
    ) -> GoalResponse:
        """Validate and reserve a new movement goal."""
        if not math.isfinite(goal_request.target_x):
            self.get_logger().warning("Rejecting non-finite target")
            return GoalResponse.REJECT

        if (
            not math.isfinite(goal_request.max_speed)
            or goal_request.max_speed <= 0.0
        ):
            self.get_logger().warning("Rejecting invalid max_speed")
            return GoalResponse.REJECT

        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warning(
                    "Rejecting goal because another goal is active"
                )
                return GoalResponse.REJECT

            self._goal_active = True

        return GoalResponse.ACCEPT

    def _accept_cancel(self, _goal_handle) -> CancelResponse:
        """Accept cancellation of the active goal."""
        self.get_logger().info("Accepting cancel request")
        return CancelResponse.ACCEPT

    def _execute_goal(
        self,
        goal_handle,
    ) -> MoveToPosition.Result:
        """Move toward the target until completed or canceled."""
        target_x = float(goal_handle.request.target_x)
        max_speed = float(goal_handle.request.max_speed)
        result = MoveToPosition.Result()

        self.get_logger().info(
            "Moving from %.2f to %.2f at max speed %.2f"
            % (self._current_x, target_x, max_speed)
        )

        try:
            while not math.isclose(
                self._current_x,
                target_x,
                abs_tol=1e-9,
            ):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()

                    result.success = False
                    result.final_x = self._current_x
                    result.message = "Goal canceled"

                    self.get_logger().info(
                        "Goal canceled at x=%.2f"
                        % self._current_x
                    )
                    return result

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
        finally:
            with self._goal_lock:
                self._goal_active = False


def main(args: list[str] | None = None) -> None:
    """Run the move Action Server until interrupted."""
    rclpy.init(args=args)
    node = MoveActionServer()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
