"""Launch the point robot publisher, subscriber, and TF broadcaster."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition


def generate_launch_description() -> LaunchDescription:
    """Create the point robot launch description."""
    package_share = Path(
        get_package_share_directory("point_robot_ros")
    )
    urdf_path = package_share / "urdf" / "point_robot.urdf"
    rviz_config_path = package_share / "rviz" / "point_robot.rviz"
    robot_description = urdf_path.read_text(encoding="utf-8")
    initial_x = LaunchConfiguration("initial_x")
    velocity_x = LaunchConfiguration("velocity_x")
    timer_period = LaunchConfiguration("timer_period")
    use_rviz = LaunchConfiguration("use_rviz")
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

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            str(rviz_config_path),
        ],
        condition=IfCondition(use_rviz),
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
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Start RViz with the saved configuration",
            ),
            position_publisher,
            position_subscriber,
            position_tf_broadcaster,
            robot_state_publisher,
            rviz,
        ]
    )
