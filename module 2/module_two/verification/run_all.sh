#!/usr/bin/env bash
# Run from any directory in Ubuntu 22.04 / ROS2 Humble after colcon build.
set -eo pipefail
module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH
source /opt/ros/humble/setup.bash
source "$module_dir/ros2_ws/install/setup.bash"
export ROS_LOCALHOST_ONLY=1
bash "$module_dir/verification/run_package_tests.sh"
python3 "$module_dir/experiments/run_experiments.py"
ROS_DOMAIN_ID=73 python3 "$module_dir/verification/check_ros_pipeline.py" --require-installed-match
ROS_DOMAIN_ID=74 ros2 run quality_navigation_demo integration_runner \
  --output "$module_dir/results/ros_navigation" --require-installed
