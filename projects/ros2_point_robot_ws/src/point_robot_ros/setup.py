import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'point_robot_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "urdf"),
            glob("urdf/*.urdf"),
        ),
        (
            os.path.join("share", package_name, "rviz"),
            glob("rviz/*.rviz"),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zxd',
    maintainer_email='258012zhou@gmail.com',
    description='ROS 2 learning package for a simulated point robot.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'position_publisher = point_robot_ros.position_publisher:main',
            'position_subscriber = point_robot_ros.position_subscriber:main',
            'reset_client = point_robot_ros.reset_client:main',
            'move_action_server = point_robot_ros.move_action_server:main',
            'move_action_client = point_robot_ros.move_action_client:main',
            'position_tf_broadcaster = point_robot_ros.position_tf_broadcaster:main',
            'camera_pid_controller = point_robot_ros.camera_pid_controller:main',
            'wheel_encoder_publisher = point_robot_ros.wheel_encoder_publisher:main',
            'odometry_node = point_robot_ros.odometry_node:main',
            'synthetic_camera_publisher = point_robot_ros.synthetic_camera_publisher:main',
            'synthetic_lidar_publisher = point_robot_ros.synthetic_lidar_publisher:main',
            'state_estimation_demo = point_robot_ros.state_estimation_demo:main',
            'safety_node = point_robot_ros.safety_node:main',
            'safe_position_simulator = point_robot_ros.safe_position_simulator:main',
        ],
    },
)
