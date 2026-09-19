#!/usr/bin/env bash
set -eo pipefail
module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH
source /opt/ros/humble/setup.bash
source "$module_dir/ros2_ws/install/setup.bash"
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=75
mkdir -p "$module_dir/results/verification"
{
  cat /etc/os-release
  uname -sr
  python3 --version
  dpkg-query -W ros-humble-rclcpp ros-humble-navigation2 ros-humble-nav2-costmap-2d \
    ros-humble-nav2-navfn-planner ros-humble-dwb-core ros-humble-cv-bridge
} > "$module_dir/results/verification/environment.txt"
python3 -m pytest "$module_dir/tests" -q \
  --junitxml="$module_dir/results/verification/python_tests.xml"
python3 -m pytest "$module_dir/ros2_ws/src/quality_navigation_demo/test" -q \
  --junitxml="$module_dir/results/verification/navigation_report_tests.xml"
cd "$module_dir/ros2_ws"
package_status=0
colcon test --executor sequential --return-code-on-test-failure \
  --event-handlers console_direct+ || package_status=$?
colcon test-result --verbose | tee "$module_dir/results/verification/colcon_test_result.txt"
exit "$package_status"
