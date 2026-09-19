cmake_minimum_required(VERSION 3.8)
project(quality_nav2_layer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(quality_nav2_layer SHARED src/quality_cost_layer.cpp)
target_include_directories(quality_nav2_layer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(quality_nav2_layer
  nav2_costmap_2d nav2_msgs pluginlib rclcpp rclcpp_lifecycle tf2 tf2_ros)
pluginlib_export_plugin_description_file(nav2_costmap_2d quality_layer_plugins.xml)

install(TARGETS quality_nav2_layer EXPORT export_quality_nav2_layer
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_cost_semantics test/test_cost_semantics.cpp)
  target_include_directories(test_cost_semantics PRIVATE include)
  ament_target_dependencies(test_cost_semantics nav2_msgs)
endif()

ament_export_targets(export_quality_nav2_layer HAS_LIBRARY_TARGET)
ament_export_dependencies(nav2_costmap_2d nav2_msgs pluginlib rclcpp rclcpp_lifecycle tf2 tf2_ros)
ament_package()
