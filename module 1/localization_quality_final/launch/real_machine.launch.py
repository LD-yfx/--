"""Launch the quality evaluator against configurable real-robot topics."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    package_share = get_package_share_directory("localization_quality")
    default_params = os.path.join(
        package_share, "config", "quality_params.yaml")
    arguments = [
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("point_cloud_topic", default_value="/points"),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument(
            "compressed_image_topic",
            default_value="/camera/image_raw/compressed"),
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument(
            "visual_localization_topic",
            default_value="/visual_localization/odom"),
        DeclareLaunchArgument(
            "laser_localization_topic",
            default_value="/laser_localization/odom"),
        DeclareLaunchArgument("image_input", default_value="raw"),
        DeclareLaunchArgument("laser_input", default_value="scan"),
        DeclareLaunchArgument("odom_required", default_value="true"),
        DeclareLaunchArgument("target_frame", default_value="odom"),
        # The Gazebo-trained calibrator is deliberately opt-in on hardware.
        DeclareLaunchArgument("mlp_enabled", default_value="false"),
        DeclareLaunchArgument("model_path", default_value=""),
        DeclareLaunchArgument("output_csv", default_value=""),
    ]
    evaluator = Node(
        package="localization_quality",
        executable="realtime_evaluator",
        name="realtime_evaluator",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": False,
                "topics.scan": LaunchConfiguration("scan_topic"),
                "topics.point_cloud": LaunchConfiguration(
                    "point_cloud_topic"),
                "topics.image": LaunchConfiguration("image_topic"),
                "topics.compressed_image": LaunchConfiguration(
                    "compressed_image_topic"),
                "topics.odom": LaunchConfiguration("odom_topic"),
                "topics.visual_localization": LaunchConfiguration(
                    "visual_localization_topic"),
                "topics.laser_localization": LaunchConfiguration(
                    "laser_localization_topic"),
                "input.image": LaunchConfiguration("image_input"),
                "input.laser": LaunchConfiguration("laser_input"),
                "input.odom_required": ParameterValue(
                    LaunchConfiguration("odom_required"), value_type=bool),
                "frames.target": LaunchConfiguration("target_frame"),
                "fusion.mlp_enabled": ParameterValue(
                    LaunchConfiguration("mlp_enabled"), value_type=bool),
                "fusion.model_path": LaunchConfiguration("model_path"),
                "output_csv": LaunchConfiguration("output_csv"),
            },
        ],
    )
    return LaunchDescription(arguments + [evaluator])
