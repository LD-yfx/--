from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config=os.path.join(get_package_share_directory('quality_aware_navigation'),'config','quality_cost.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time',default_value='false'),
        DeclareLaunchArgument('params_file',default_value=config),
        Node(package='quality_aware_navigation',executable='quality_cost_node',output='screen',
             parameters=[LaunchConfiguration('params_file'),
                         {'use_sim_time':ParameterValue(LaunchConfiguration('use_sim_time'),value_type=bool)}]),
    ])
