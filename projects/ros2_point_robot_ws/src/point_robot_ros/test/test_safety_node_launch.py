"""Integration test for the ROS 2 velocity safety node."""

import time
import unittest

from geometry_msgs.msg import Twist
import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy


@pytest.mark.launch_test
def generate_test_description():
    """Launch the safety node with short deterministic test settings."""
    safety_node = launch_ros.actions.Node(
        package="point_robot_ros",
        executable="safety_node",
        name="safety_node_integration_test",
        output="screen",
        parameters=[
            {
                "max_linear_speed": 1.0,
                "command_timeout": 0.3,
                "publish_period": 0.02,
            }
        ],
    )

    return launch.LaunchDescription(
        [
            safety_node,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestSafetyNode(unittest.TestCase):
    """Verify command limiting and timeout stopping through ROS 2."""

    @classmethod
    def setUpClass(cls):
        """Initialize ROS 2 before running the test."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shut down ROS 2 after running the test."""
        rclpy.shutdown()

    def setUp(self):
        """Create a temporary command publisher and output subscriber."""
        self.node = rclpy.create_node("safety_node_integration_client")
        self.received_commands: list[float] = []
        self.publisher = self.node.create_publisher(
            Twist,
            "/point_robot/cmd_vel_raw",
            10,
        )
        self.subscription = self.node.create_subscription(
            Twist,
            "/point_robot/cmd_vel_safe",
            self._record_command,
            10,
        )

    def tearDown(self):
        """Destroy the temporary test node."""
        self.node.destroy_publisher(self.publisher)
        self.node.destroy_subscription(self.subscription)
        self.node.destroy_node()

    def _record_command(self, message: Twist) -> None:
        """Store the safe linear velocity from each received message."""
        self.received_commands.append(float(message.linear.x))

    def test_limit_then_timeout_stop(self):
        """Limit an excessive command, then stop after command timeout."""
        raw_command = Twist()
        raw_command.linear.x = 2.0

        limit_deadline = time.monotonic() + 5.0
        while (
            not any(value == pytest.approx(1.0)
                    for value in self.received_commands)
            and time.monotonic() < limit_deadline
        ):
            self.publisher.publish(raw_command)
            rclpy.spin_once(self.node, timeout_sec=0.05)

        self.assertTrue(
            any(
                value == pytest.approx(1.0)
                for value in self.received_commands
            ),
            "Expected excessive command to be limited to 1.0",
        )

        self.received_commands.clear()
        stop_deadline = time.monotonic() + 2.0
        while (
            not any(value == pytest.approx(0.0)
                    for value in self.received_commands)
            and time.monotonic() < stop_deadline
        ):
            rclpy.spin_once(self.node, timeout_sec=0.05)

        self.assertTrue(
            any(
                value == pytest.approx(0.0)
                for value in self.received_commands
            ),
            "Expected stale command to produce a zero safe velocity",
        )
