#!/usr/bin/env bash
set -eo pipefail
module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
research_workspace="${SITP_RESEARCH_WS:-$HOME/sitp_quality_research/ros2_ws}"
mkdir -p "$research_workspace/src"
# Windows source remains authoritative; no --delete and no generated build migration.
rsync -a --exclude __pycache__ --exclude .pytest_cache \
  "$module_dir/ros2_ws/src/" "$research_workspace/src/"
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
cd "$research_workspace"
export CMAKE_BUILD_PARALLEL_LEVEL=4
colcon build --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  --event-handlers console_direct+ "$@"
