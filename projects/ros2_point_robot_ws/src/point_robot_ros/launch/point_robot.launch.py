"""Launch the point robot publisher, subscriber, and TF broadcaster."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Create the point robot launch description."""
    initial_x = LaunchConfiguration("initial_x")
    velocity_x = LaunchConfiguration("velocity_x")
    timer_period = LaunchConfiguration("timer_period")
    position_publisher = Node(
        package="point_robot_ros",
        executable="position_publisher",
        name="position_publisher",
        output="screen",
        parameters=[
            {
                "initial_x": ParameterValue(
                    initial_x,
                    value_type=float,
                ),
                "velocity_x": ParameterValue(
                    velocity_x,
                    value_type=float,
                ),
                "timer_period": ParameterValue(
                    timer_period,
                    value_type=float,
                ),
            }
        ],
    )

    position_subscriber = Node(
        package="point_robot_ros",
        executable="position_subscriber",
        name="position_subscriber",
        output="screen",
    )

    position_tf_broadcaster = Node(
        package="point_robot_ros",
        executable="position_tf_broadcaster",
        name="position_tf_broadcaster",
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "initial_x",
                default_value="0.0",
                description="Initial x position",
            ),
            DeclareLaunchArgument(
                "velocity_x",
                default_value="0.5",
                description="Velocity along the x axis",
            ),
            DeclareLaunchArgument(
                "timer_period",
                default_value="0.1",
                description="Publishing period in seconds",
            ),
            position_publisher,
            position_subscriber,
            position_tf_broadcaster,
        ]
    )
