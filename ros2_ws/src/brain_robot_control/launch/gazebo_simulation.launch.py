import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('brain_robot_control')
    param_file = os.path.join(pkg_share, 'config', 'controller_params.yaml')
    bridge_params = os.path.join(pkg_share, 'config', 'ros_gz_bridge.yaml')
    world_file = os.path.join(pkg_share, 'worlds', 'neuromorphic_arena.world')
    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    # TurtleBot3 World Simulation Launch
    tb3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_dir, 'launch', 'turtlebot3_world.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    # Custom ROS-Gazebo Bridge (Ensures /cmd_vel Twist & /scan LaserScan bridge directly)
    custom_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output='screen',
    )

    # Neuromorphic Brain Controller Node
    brain_node = Node(
        package='brain_robot_control',
        executable='neuromorphic_node',
        name='neuromorphic_brain_node',
        output='screen',
        parameters=[param_file]
    )

    return LaunchDescription([
        tb3_launch,
        custom_bridge_node,
        brain_node
    ])
