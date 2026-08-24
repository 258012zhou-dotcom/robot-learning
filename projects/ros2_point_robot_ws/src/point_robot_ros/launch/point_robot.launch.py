"""Launch the point robot publisher, subscriber, and TF broadcaster."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    target_yaw = LaunchConfiguration("target_yaw")
    kp = LaunchConfiguration("kp")
    ki = LaunchConfiguration("ki")
    kd = LaunchConfiguration("kd")
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

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        condition=LaunchConfigurationEquals(
            "joint_control_mode",
            "publisher",
        ),
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
        condition=LaunchConfigurationEquals(
            "joint_control_mode",
            "gui",
        ),
    )

    camera_pid_controller = Node(
        package="point_robot_ros",
        executable="camera_pid_controller",
        name="camera_pid_controller",
        output="screen",
        parameters=[
            {
                "target_yaw": ParameterValue(
                    target_yaw,
                    value_type=float,
                ),
                "kp": ParameterValue(
                    kp,
                    value_type=float,
                ),
                "ki": ParameterValue(
                    ki,
                    value_type=float,
                ),
                "kd": ParameterValue(
                    kd,
                    value_type=float,
                ),
            }
        ],
        condition=LaunchConfigurationEquals(
            "joint_control_mode",
            "pid",
        ),
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
            DeclareLaunchArgument(
                "joint_control_mode",
                default_value="publisher",
                description="Joint source: publisher, gui, or pid",
                choices=["publisher", "gui", "pid"],
            ),
            DeclareLaunchArgument(
                "target_yaw",
                default_value="1.0",
                description="Target camera yaw in radians",
            ),
            DeclareLaunchArgument(
                "kp",
                default_value="2.0",
                description="PID proportional gain",
            ),
            DeclareLaunchArgument(
                "ki",
                default_value="0.0",
                description="PID integral gain",
            ),
            DeclareLaunchArgument(
                "kd",
                default_value="0.0",
                description="PID derivative gain",
            ),
            position_publisher,
            position_subscriber,
            position_tf_broadcaster,
            robot_state_publisher,
            joint_state_publisher,
            joint_state_publisher_gui,
            camera_pid_controller,
            rviz,
        ]
    )
