#!/usr/bin/env bash
set -euo pipefail
MODULE_TWO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "$MODULE_TWO_DIR/ros2_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-74}"
# Source overlay permits iterative Python development without altering generated
# install artifacts; all ROS dependencies and C++ plugins are from the build.
export PYTHONPATH="$MODULE_TWO_DIR/ros2_ws/src/quality_navigation_demo:$MODULE_TWO_DIR/ros2_ws/src/quality_aware_navigation:${PYTHONPATH:-}"
exec python3 -m quality_navigation_demo.runner --output "$MODULE_TWO_DIR/results/ros_navigation" "$@"
