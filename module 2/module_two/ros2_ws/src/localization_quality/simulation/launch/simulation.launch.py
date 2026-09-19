"""Launch one deterministic Gazebo data-collection run."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def setup(context):
    package_share = get_package_share_directory("localization_quality")
    gazebo_share = get_package_share_directory("gazebo_ros")
    world = LaunchConfiguration("world").perform(context)
    scene = LaunchConfiguration("scene").perform(context)
    route = LaunchConfiguration("route").perform(context)
    lighting = LaunchConfiguration("lighting").perform(context)
    geometry = LaunchConfiguration("geometry_complexity").perform(context)
    fault = LaunchConfiguration("fault").perform(context)
    seed = int(LaunchConfiguration("seed").perform(context))
    speed = float(LaunchConfiguration("speed_mps").perform(context))
    start = {
        "corridor": (-16.0, 0.0),
        "hall": (-10.0, -10.0),
        "outdoor": (-12.0, -10.0),
    }[scene]
    robot_file = os.path.join(
        package_share, "simulation", "robot", "sensor_robot.urdf.xacro")
    description = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", robot_file]),
        value_type=str,
    )
    common = {
        "use_sim_time": True,
        "scene": scene,
        "route": route,
        "lighting": lighting,
        "geometry_complexity": geometry,
        "fault": fault,
        "seed": seed,
        "speed_mps": speed,
    }
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, "launch", "gazebo.launch.py")),
            launch_arguments={
                "world": world, "verbose": "false", "gui": "false",
            }.items(),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"use_sim_time": True, "robot_description": description}],
        ),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-entity", "localization_quality_robot",
                "-topic", "robot_description",
                "-x", str(start[0]), "-y", str(start[1]), "-z", "0.15",
            ],
            output="screen",
        ),
        Node(
            package="localization_quality",
            executable="simulation_sensor_pipeline.py",
            parameters=[{"use_sim_time": True, "fault": fault}],
        ),
        Node(
            package="localization_quality",
            executable="simulation_localization_proxy.py",
            parameters=[common],
        ),
        Node(
            package="localization_quality",
            executable="simulation_route_driver.py",
            parameters=[{
                "use_sim_time": True,
                "scene": scene,
                "route": route,
                "speed_mps": speed,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world"),
        DeclareLaunchArgument("scene"),
        DeclareLaunchArgument("lighting"),
        DeclareLaunchArgument("geometry_complexity"),
        DeclareLaunchArgument("fault"),
        DeclareLaunchArgument("route"),
        DeclareLaunchArgument("seed"),
        DeclareLaunchArgument("speed_mps"),
        OpaqueFunction(function=setup),
    ])
