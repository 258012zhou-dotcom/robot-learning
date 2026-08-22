"""Integration test for the point robot position topic."""

import time
import unittest

from geometry_msgs.msg import Point
import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy


@pytest.mark.launch_test
def generate_test_description():
    """Launch the position publisher for integration testing."""
    publisher = launch_ros.actions.Node(
        package="point_robot_ros",
        executable="position_publisher",
        name="position_publisher_integration_test",
        output="screen",
        parameters=[
            {
                "initial_x": 0.0,
                "velocity_x": 1.0,
                "timer_period": 0.05,
            }
        ],
    )

    return launch.LaunchDescription(
        [
            publisher,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestPositionTopic(unittest.TestCase):
    """Verify position messages transmitted through ROS 2."""

    @classmethod
    def setUpClass(cls):
        """Initialize ROS 2 before running the tests."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shut down ROS 2 after running the tests."""
        rclpy.shutdown()

    def setUp(self):
        """Create a temporary subscriber node."""
        self.node = rclpy.create_node("position_topic_integration_test")

    def tearDown(self):
        """Destroy the temporary subscriber node."""
        self.node.destroy_node()

    def test_position_messages_increase(self):
        """Receive position messages and verify their motion."""
        received_messages: list[Point] = []

        subscription = self.node.create_subscription(
            Point,
            "/point_robot/position",
            received_messages.append,
            10,
        )

        try:
            deadline = time.monotonic() + 5.0

            while (
                len(received_messages) < 3
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(self.node, timeout_sec=0.1)

            self.assertGreaterEqual(
                len(received_messages),
                3,
                "Expected at least three position messages",
            )

            self.assertTrue(
                all(message.y == 0.0 for message in received_messages)
            )

            self.assertGreater(
                received_messages[-1].x,
                received_messages[0].x,
            )

            step = received_messages[1].x - received_messages[0].x
            self.assertAlmostEqual(step, 0.05, places=6)
        finally:
            self.node.destroy_subscription(subscription)
