#!/usr/bin/env bash
set -eo pipefail

# Use installed production packages; do not inherit an unrelated source overlay.
verification_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="${verification_dir}/../ros2_ws"
unset PYTHONPATH
source /opt/ros/humble/setup.bash
source "${workspace_dir}/install/setup.bash"
python3 -c 'import rclpy; print("ROS2 Python imports ready", flush=True)'
exec python3 "${verification_dir}/check_ros_pipeline.py" "$@"
