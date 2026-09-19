cmake_minimum_required(VERSION 3.8)
project(localization_quality)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(quality_navigation_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_library(localization_quality_core
  src/quality_core.cpp
  src/quality_advanced.cpp
)
target_include_directories(localization_quality_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(localization_quality_core ${OpenCV_LIBS})

add_executable(offline_evaluator src/offline_evaluator.cpp)
target_link_libraries(offline_evaluator localization_quality_core)
ament_target_dependencies(offline_evaluator
  ament_index_cpp
  rclcpp
  sensor_msgs
  nav_msgs
  rosbag2_cpp
  cv_bridge
)

add_executable(realtime_evaluator src/realtime_evaluator.cpp)
target_link_libraries(realtime_evaluator localization_quality_core)
ament_target_dependencies(realtime_evaluator
  ament_index_cpp
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
  quality_navigation_msgs
  diagnostic_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
)

install(
  TARGETS localization_quality_core offline_evaluator realtime_evaluator
  EXPORT localization_quality_core_export
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY docs/ DESTINATION share/${PROJECT_NAME}/docs)
install(
  DIRECTORY simulation/
  DESTINATION share/${PROJECT_NAME}/simulation
  PATTERN "__pycache__" EXCLUDE
  PATTERN "launch" EXCLUDE
)
install(
  DIRECTORY simulation/launch/
  DESTINATION share/${PROJECT_NAME}/launch
  PATTERN "__pycache__" EXCLUDE
)
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  PATTERN "__pycache__" EXCLUDE
)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
install(FILES README.md UPSTREAM_README.md LICENSE CHANGELOG.md DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/validate_results.py
  scripts/generate_quality_map.py
  scripts/generate_quality_grid.py
  scripts/generate_formal_quality_maps.py
  scripts/validate_quality_map_collection.py
  scripts/prepare_dataset.py
  scripts/build_training_dataset.py
  scripts/extract_training_rows.py
  scripts/train_context_mlp.py
  scripts/evaluate_ablations.py
  scripts/generate_sim_worlds.py
  scripts/generate_bootstrap_training_data.py
  scripts/validate_module_one.py
  scripts/compare_offline_realtime.py
  scripts/summarize_formal_dataset.py
  scripts/summarize_real_bag_smoke.py
  scripts/create_hash_manifest.py
  scripts/collect_sim_dataset.py
  scripts/simulation_sensor_pipeline.py
  scripts/simulation_localization_proxy.py
  scripts/simulation_route_driver.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/reproduce_archive.sh
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  ament_add_gtest(
    test_quality_core
    test/test_quality_core.cpp
    test/test_advanced.cpp
    test/test_causal_sync.cpp
  )
  target_link_libraries(test_quality_core localization_quality_core)
  target_include_directories(test_quality_core PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(
    test_quality_core PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data"
    BOOTSTRAP_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/bootstrap_context_mlp.yaml"
    FORMAL_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/context_mlp_formal.yaml"
  )
  ament_add_gtest(test_quality_statistics test/test_quality_statistics.cpp)
  target_link_libraries(test_quality_statistics localization_quality_core)
  ament_target_dependencies(test_quality_statistics quality_navigation_msgs)
  add_test(
    NAME test_python_pipeline
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_python_pipeline.py
  )
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/formal_gazebo" AND
     EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/training_report.json")
    add_test(
      NAME test_formal_dataset_matrix
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_dataset.py
        validate-matrix
        ${CMAKE_CURRENT_SOURCE_DIR}/data/formal_gazebo
    )
    add_test(
      NAME test_formal_acceptance
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_module_one.py
        --training-report
        ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/training_report.json
        --runtime-csv
        ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/runtime_normal.csv
        --ablation-report
        ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/ablation_report.json
        --quality-map-report
        ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/quality_map_summary.json
        --expected-sensor-frames 1329
        --output ${CMAKE_CURRENT_BINARY_DIR}/formal_acceptance_ctest.json
    )
    add_test(
      NAME test_offline_realtime_consistency
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_offline_realtime.py
        ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/runtime_normal.csv
        ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/realtime_playback.csv
        --expected-sensor-frames 1329
        --output ${CMAKE_CURRENT_BINARY_DIR}/offline_realtime_ctest.json
    )
    if(EXISTS
        "${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/quality_map_summary.json")
      add_test(
        NAME test_formal_quality_map_collection
        COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_quality_map_collection.py
          ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/quality_maps
          ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/quality_map_summary.json
          ${CMAKE_CURRENT_SOURCE_DIR}/results/formal_training/quality_map_summary.csv
          --expected-runs 27
      )
    endif()
  endif()
endif()

ament_export_targets(localization_quality_core_export HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(quality_navigation_msgs)
ament_package()
