#pragma once

#include "localization_quality/quality_core.hpp"

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "rclcpp/rclcpp.hpp"

#include <string>

namespace localization_quality
{

inline std::string resolve_package_model_path(const std::string & value)
{
  const std::string prefix = "package://localization_quality/";
  if (value.compare(0, prefix.size(), prefix) != 0) {
    return value;
  }
  return ament_index_cpp::get_package_share_directory("localization_quality") +
         "/" + value.substr(prefix.size());
}

inline QualityConfig declare_quality_parameters(rclcpp::Node & node)
{
  QualityConfig config;
  config.laser_min_range_m = node.declare_parameter<double>(
    "quality.laser.min_range_m", config.laser_min_range_m);
  config.laser_max_range_m = node.declare_parameter<double>(
    "quality.laser.max_range_m", config.laser_max_range_m);
  config.laser_close_range_m = node.declare_parameter<double>(
    "quality.laser.close_range_m", config.laser_close_range_m);
  config.laser_occlusion_start_ratio = node.declare_parameter<double>(
    "quality.laser.occlusion_start_ratio", config.laser_occlusion_start_ratio);
  config.laser_occlusion_floor = node.declare_parameter<double>(
    "quality.laser.occlusion_floor", config.laser_occlusion_floor);
  config.laser_coverage_sectors = node.declare_parameter<int>(
    "quality.laser.coverage_sectors", config.laser_coverage_sectors);
  config.laser_coverage_weight = node.declare_parameter<double>(
    "quality.laser.coverage_weight", config.laser_coverage_weight);
  config.laser_entropy_weight = node.declare_parameter<double>(
    "quality.laser.entropy_weight", config.laser_entropy_weight);
  config.laser_geometry_weight = node.declare_parameter<double>(
    "quality.laser.geometry_weight", config.laser_geometry_weight);
  config.laser_abrupt_jump_m = node.declare_parameter<double>(
    "quality.laser.abrupt_jump_m", config.laser_abrupt_jump_m);

  config.visual_blur_bad = node.declare_parameter<double>(
    "quality.visual.blur_bad", config.visual_blur_bad);
  config.visual_blur_good = node.declare_parameter<double>(
    "quality.visual.blur_good", config.visual_blur_good);
  config.visual_brightness_bad_low = node.declare_parameter<double>(
    "quality.visual.brightness_bad_low", config.visual_brightness_bad_low);
  config.visual_brightness_ideal_low = node.declare_parameter<double>(
    "quality.visual.brightness_ideal_low", config.visual_brightness_ideal_low);
  config.visual_brightness_ideal_high = node.declare_parameter<double>(
    "quality.visual.brightness_ideal_high", config.visual_brightness_ideal_high);
  config.visual_brightness_bad_high = node.declare_parameter<double>(
    "quality.visual.brightness_bad_high", config.visual_brightness_bad_high);
  config.visual_contrast_bad = node.declare_parameter<double>(
    "quality.visual.contrast_bad", config.visual_contrast_bad);
  config.visual_contrast_good = node.declare_parameter<double>(
    "quality.visual.contrast_good", config.visual_contrast_good);
  config.visual_blur_weight = node.declare_parameter<double>(
    "quality.visual.blur_weight", config.visual_blur_weight);
  config.visual_brightness_weight = node.declare_parameter<double>(
    "quality.visual.brightness_weight", config.visual_brightness_weight);
  config.visual_contrast_weight = node.declare_parameter<double>(
    "quality.visual.contrast_weight", config.visual_contrast_weight);
  config.visual_exposure_weight = node.declare_parameter<double>(
    "quality.visual.exposure_weight", config.visual_exposure_weight);
  config.visual_edge_weight = node.declare_parameter<double>(
    "quality.visual.edge_weight", config.visual_edge_weight);
  config.visual_max_features = node.declare_parameter<int>(
    "quality.visual.max_features", config.visual_max_features);
  config.visual_min_features = node.declare_parameter<int>(
    "quality.visual.min_features", config.visual_min_features);
  config.visual_underexposed_threshold = node.declare_parameter<double>(
    "quality.visual.underexposed_threshold", config.visual_underexposed_threshold);
  config.visual_overexposed_threshold = node.declare_parameter<double>(
    "quality.visual.overexposed_threshold", config.visual_overexposed_threshold);

  config.failure_enter_threshold = node.declare_parameter<double>(
    "fusion.failure_enter_threshold", config.failure_enter_threshold);
  config.failure_recovery_threshold = node.declare_parameter<double>(
    "fusion.failure_recovery_threshold", config.failure_recovery_threshold);
  config.failed_sensor_weight = node.declare_parameter<double>(
    "fusion.failed_sensor_weight", config.failed_sensor_weight);
  config.quality_ema_alpha = node.declare_parameter<double>(
    "fusion.quality_ema_alpha", config.quality_ema_alpha);
  config.visual_calibration_domain_guard_enabled = node.declare_parameter<bool>(
    "fusion.visual_calibration_domain_guard.enabled",
    config.visual_calibration_domain_guard_enabled);
  config.visual_calibration_domain_guard_raw_min = node.declare_parameter<double>(
    "fusion.visual_calibration_domain_guard.raw_min",
    config.visual_calibration_domain_guard_raw_min);
  config.visual_calibration_domain_guard_calibrated_max =
    node.declare_parameter<double>(
    "fusion.visual_calibration_domain_guard.calibrated_max",
    config.visual_calibration_domain_guard_calibrated_max);
  config.scene_classifier_enabled = node.declare_parameter<bool>(
    "context.scene_classifier_enabled", config.scene_classifier_enabled);
  config.mlp_enabled = node.declare_parameter<bool>(
    "fusion.mlp_enabled", config.mlp_enabled);
  config.fusion_strategy = node.declare_parameter<std::string>(
    "fusion.strategy", config.fusion_strategy);
  config.context_model_path = resolve_package_model_path(
    node.declare_parameter<std::string>(
      "fusion.model_path", config.context_model_path));
  validate_config(config);
  return config;
}

}  // namespace localization_quality
