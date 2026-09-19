"""Gazebo sensors + encoder odometry + independent AMCL + real module one/Nav2.

No old ground-truth-noise localization proxy, route driver, or ideal simulator.
World, robot, map, and nominal initialization are explicit run inputs.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def typed(name,kind):
    return ParameterValue(LaunchConfiguration(name),value_type=kind)


def planner_parameters(source_file):
    # The geometric baseline must not gain quality costs even when the quality
    # source stalls. Keep its sensing/computation load, disable only this layer.
    return RewrittenYaml(source_file=source_file,param_rewrites={
        'global_costmap.global_costmap.ros__parameters.quality_layer.enabled':
            PythonExpression(["'",LaunchConfiguration('cost_mode'),"' != 'geometric'"]),
    },convert_types=True)


def generate_launch_description():
    share=get_package_share_directory('quality_localization_benchmark')
    module_one=get_package_share_directory('localization_quality')
    demo=get_package_share_directory('quality_navigation_demo')
    gazebo=get_package_share_directory('gazebo_ros')
    params=os.path.join(share,'config','nav2_localization.yaml')
    quality_params=os.path.join(module_one,'config','quality_params.yaml')
    arguments=[DeclareLaunchArgument('world'),DeclareLaunchArgument('map'),
               DeclareLaunchArgument('robot')]
    for name,default in {
        'spawn_x':'-4.6','spawn_y':'0.0','spawn_yaw':'0.0',
        'initial_x':'-4.4','initial_y':'0.1','initial_yaw':'0.05',
        'initial_sigma_xy':'0.3','initial_sigma_yaw':'0.15',
        'quality_map_resolution':'0.2','quality_map_width':'60','quality_map_height':'40',
        'quality_map_origin_x':'-6.0','quality_map_origin_y':'-4.0',
        'statistics_window_sec':'10.0','cost_mode':'full','module_one_csv':'',
        'zones_json':'','sensor_seed':'-1','gazebo_seed':'42',
    }.items():arguments.append(DeclareLaunchArgument(name,default_value=default))
    robot_description=ParameterValue(Command([
        FindExecutable(name='xacro'),' ',LaunchConfiguration('robot')]),value_type=str)
    nodes=[
        Node(package='quality_localization_audit',executable='tf_authority_audit',
             name='tf_authority_audit',parameters=[{'use_sim_time':True}],output='screen'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(gazebo,'launch','gazebo.launch.py')),
            launch_arguments={'world':LaunchConfiguration('world'),'gui':'false',
                              'verbose':'false','seed':LaunchConfiguration('gazebo_seed'),
                              'params_file':params,'server_required':'true'}.items()),
        Node(package='robot_state_publisher',executable='robot_state_publisher',
             parameters=[{'use_sim_time':True,'robot_description':robot_description}],output='screen'),
        Node(package='gazebo_ros',executable='spawn_entity.py',output='screen',
             parameters=[{'use_sim_time':True}],arguments=[
            '-entity','research_robot','-topic','robot_description',
            '-x',LaunchConfiguration('spawn_x'),'-y',LaunchConfiguration('spawn_y'),
            '-Y',LaunchConfiguration('spawn_yaw'),'-z','0.05']),
        Node(package='quality_localization_benchmark',executable='sensor_bridge',
             name='sensor_bridge',parameters=[{
                 'use_sim_time':True,'zones_json':LaunchConfiguration('zones_json'),
                 'seed':typed('sensor_seed',int),
             }],output='screen'),
        Node(package='nav2_map_server',executable='map_server',name='map_server',
             parameters=[params,{'yaml_filename':LaunchConfiguration('map')}],output='screen'),
        Node(package='nav2_amcl',executable='amcl',name='amcl',parameters=[params],output='screen'),
        Node(package='quality_localization_benchmark',executable='localization_adapter',
             name='localization_adapter',parameters=[params,{
                 key:typed(key,float) for key in ('initial_x','initial_y','initial_yaw',
                                                 'initial_sigma_xy','initial_sigma_yaw')}],output='screen'),
        Node(package='localization_quality',executable='realtime_evaluator',
             name='realtime_evaluator',parameters=[quality_params,{
                 'use_sim_time':True,'topics.scan':'/scan','topics.image':'/camera/image_raw',
                 'topics.odom':'/localization/estimated_odom',
                 'topics.laser_localization':'/laser_localization/odom',
                 # No visual estimator is claimed in the AMCL baseline. Image
                 # features remain real module-one inputs, tracking is unknown.
                 'topics.visual_localization':'/unavailable_visual_localization/odom',
                 'input.image':'raw','input.laser':'scan','frames.target':'map',
                 'fusion.model_path':'','fusion.strategy':'rule','fusion.mlp_enabled':False,
                 'run_mode':'normal','map.publish_statistics':True,
                 'map.resolution':typed('quality_map_resolution',float),
                 'map.width':typed('quality_map_width',int),'map.height':typed('quality_map_height',int),
                 'map.origin_x':typed('quality_map_origin_x',float),
                 'map.origin_y':typed('quality_map_origin_y',float),
                 'map.statistics_window_sec':typed('statistics_window_sec',float),
                 'output_csv':LaunchConfiguration('module_one_csv'),
             }],output='screen'),
        Node(package='quality_aware_navigation',executable='quality_cost_node',
             name='quality_cost_node',parameters=[params,{'mode':LaunchConfiguration('cost_mode')}],output='screen'),
        Node(package='nav2_controller',executable='controller_server',name='controller_server',
             parameters=[params],output='screen'),
        Node(package='nav2_planner',executable='planner_server',name='planner_server',
             parameters=[planner_parameters(params)],output='screen'),
        Node(package='nav2_bt_navigator',executable='bt_navigator',name='bt_navigator',
             parameters=[params,{
                 'default_nav_to_pose_bt_xml':os.path.join(demo,'config','navigate.xml'),
                 'default_nav_through_poses_bt_xml':os.path.join(demo,'config','navigate_through.xml'),
             }],output='screen'),
        Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',
             name='lifecycle_manager_benchmark',parameters=[{
                 'use_sim_time':True,'autostart':True,'bond_timeout':10.0,
                 'node_names':['map_server','amcl','controller_server','planner_server','bt_navigator'],
             }],output='screen'),
    ]
    return LaunchDescription(arguments+nodes)
