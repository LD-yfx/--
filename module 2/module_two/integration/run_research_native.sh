#!/usr/bin/env bash
set -eo pipefail
research_workspace="${SITP_RESEARCH_WS:-$HOME/sitp_quality_research/ros2_ws}"
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH
source /opt/ros/humble/setup.bash
source "$research_workspace/install/setup.bash"
entrypoint="${1:-benchmark_runner}"
if [ "$#" -gt 0 ]; then shift; fi
case "$entrypoint" in
  benchmark_runner|benchmark_campaign|benchmark_plot) ;;
  *) echo "Unknown research entrypoint: $entrypoint" >&2; exit 2 ;;
esac
exec ros2 run quality_localization_benchmark "$entrypoint" "$@"
