import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    share=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    params=os.path.join(share,'config','nav2.yaml')
    return LaunchDescription([
        Node(package='quality_navigation_demo',executable='kinematic_sim',output='screen'),
        Node(package='quality_aware_navigation',executable='quality_cost_node',
             parameters=[params],output='screen'),
        Node(package='nav2_controller',executable='controller_server',name='controller_server',
             parameters=[params],output='screen'),
        Node(package='nav2_planner',executable='planner_server',name='planner_server',
             parameters=[params],output='screen'),
        Node(package='nav2_bt_navigator',executable='bt_navigator',name='bt_navigator',
             parameters=[params,{
                 'default_nav_to_pose_bt_xml':os.path.join(share,'config','navigate.xml'),
                 'default_nav_through_poses_bt_xml':os.path.join(share,'config','navigate_through.xml')}],
             output='screen'),
        Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',
             name='lifecycle_manager_navigation',output='screen',parameters=[{
                 'autostart':True,'bond_timeout':10.0,
                 'node_names':['controller_server','planner_server','bt_navigator']}]),
    ])
