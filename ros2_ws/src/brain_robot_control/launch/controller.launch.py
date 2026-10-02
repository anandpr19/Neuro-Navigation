import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('brain_robot_control')
    param_file = os.path.join(pkg_share, 'config', 'controller_params.yaml')

    return LaunchDescription([
        Node(
            package='brain_robot_control',
            executable='neuromorphic_node',
            name='neuromorphic_brain_node',
            output='screen',
            parameters=[param_file]
        )
    ])
