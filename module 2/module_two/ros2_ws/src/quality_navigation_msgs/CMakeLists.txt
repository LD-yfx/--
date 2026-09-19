cmake_minimum_required(VERSION 3.8)
project(quality_navigation_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/QualityGrid.msg"
  DEPENDENCIES std_msgs nav_msgs builtin_interfaces
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
