"""Verify the real launch wiring from raw command to simulated motion."""

from pathlib import Path
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Twist
import launch
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing.actions
import pytest
import rclpy
from std_srvs.srv import SetBool


@pytest.mark.launch_test
def generate_test_description():
    """Start the public launch file in safety mode without GUI or hardware."""
    launch_path = Path(get_package_share_directory("point_robot_ros"))
    system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(launch_path / "launch/point_robot.launch.py")),
        launch_arguments={
            "use_safety": "true",
            "timer_period": "0.02",
            "max_linear_speed": "0.5",
            # A long timeout distinguishes invalid-input stopping from expiry.
            "command_timeout": "2.0",
        }.items(),
    )
    return launch.LaunchDescription([system, launch_testing.actions.ReadyToTest()])


class TestSafeMotion(unittest.TestCase):
    """Exercise limits, invalid input, the emergency latch, and timeout."""

    @classmethod
    def setUpClass(cls):
        """Initialize the test's ROS context."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Release the ROS context."""
        rclpy.shutdown()

    def setUp(self):
        """Observe both safe commands and actual simulated positions."""
        self.node = rclpy.create_node("safe_motion_test")
        self.positions = []
        self.velocities = []
        self.node.create_subscription(
            Point, "/point_robot/position", lambda msg: self.positions.append(msg.x), 10,
        )
        self.node.create_subscription(
            Twist, "/point_robot/cmd_vel_safe",
            lambda msg: self.velocities.append(msg.linear.x), 10,
        )
        self.raw = self.node.create_publisher(Twist, "/point_robot/cmd_vel_raw", 10)
        self.emergency = self.node.create_client(SetBool, "/point_robot/set_emergency_stop")

    def tearDown(self):
        """Destroy all test publishers, subscribers, and clients."""
        self.node.destroy_node()

    def _spin_for(self, duration, command=None):
        """Receive messages; optionally send a command at approximately 20 Hz."""
        deadline = time.monotonic() + duration
        next_publish = 0.0
        while time.monotonic() < deadline:
            if command is not None and time.monotonic() >= next_publish:
                message = Twist()
                message.linear.x = command
                self.raw.publish(message)
                next_publish = time.monotonic() + 0.05
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def _assert_stopped(self):
        """Check several new positions, not a stale zero-velocity message."""
        self.positions.clear()
        self.velocities.clear()
        self._spin_for(0.25)
        self.assertGreaterEqual(len(self.positions), 5)
        self.assertGreaterEqual(len(self.velocities), 2)
        self.assertLess(max(self.positions) - min(self.positions), 1e-9)
        self.assertTrue(all(value == 0.0 for value in self.velocities))

    def _assert_moving(self, command):
        """Fresh commands must produce motion within the configured limit."""
        self.positions.clear()
        self.velocities.clear()
        self._spin_for(0.5, command=command)
        self.assertGreaterEqual(len(self.positions), 5)
        self.assertGreater(self.positions[-1] - self.positions[0], 0.05)
        self.assertTrue(all(abs(value) <= 0.5 for value in self.velocities))

    def _set_emergency(self, active):
        """Wait for an acknowledged emergency-state change with a deadline."""
        future = self.emergency.call_async(SetBool.Request(data=active))
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        self.assertTrue(future.done(), "Emergency service did not respond")
        self.assertTrue(future.result().success)
        self._spin_for(0.15)

    def test_motion_is_gated_by_safety(self):
        """Reject stale/invalid commands and never resume a pre-stop command."""
        self.assertTrue(self.emergency.wait_for_service(timeout_sec=5.0))
        deadline = time.monotonic() + 5.0
        while not self.positions and time.monotonic() < deadline:
            self._spin_for(0.1)
        self.assertTrue(self.positions, "Simulation did not publish position")
        # Catch the old bug: an independent constant-speed publisher must not run.
        self.assertEqual(self.node.count_publishers("/point_robot/position"), 1)
        self.assertEqual(self.node.count_subscribers("/point_robot/cmd_vel_raw"), 1)
        self.assertEqual(self.node.count_subscribers("/point_robot/cmd_vel_safe"), 2)
        self._assert_stopped()
        self._assert_moving(2.0)

        invalid_sent = time.monotonic()
        self._spin_for(0.2, command=float("nan"))
        self._assert_stopped()
        self.assertLess(time.monotonic() - invalid_sent, 1.0)

        self._assert_moving(0.4)
        self._set_emergency(True)
        self._spin_for(0.2, command=0.4)
        self._assert_stopped()
        self._set_emergency(False)
        self._assert_stopped()

        self._assert_moving(0.4)
        self._spin_for(2.3)  # Stop transmitting; allow the watchdog to expire.
        self._assert_stopped()
