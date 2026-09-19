#!/usr/bin/env bash
set -euo pipefail
MODULE_TWO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "$MODULE_TWO_DIR/ros2_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-74}"
exec ros2 run quality_navigation_demo integration_runner \
  --require-installed --output "$MODULE_TWO_DIR/results/ros_navigation" "$@"
