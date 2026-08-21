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
        ],
    },
)
