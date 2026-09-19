#include "localization_quality/quality_core.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace localization_quality
{
namespace
{

double clamp(double value, double low, double high)
{
  return std::max(low, std::min(value, high));
}

double clamp01(double value)
{
  return clamp(value, 0.0, 1.0);
}

double ramp(double value, double bad, double good)
{
  if (good <= bad) {
    return 0.0;
  }
  return clamp01((value - bad) / (good - bad));
}

double brightness_score(double value, const QualityConfig & config)
{
  if (value <= config.visual_brightness_bad_low ||
    value >= config.visual_brightness_bad_high)
  {
    return 0.0;
  }
  if (value < config.visual_brightness_ideal_low) {
    return ramp(
      value, config.visual_brightness_bad_low,
      config.visual_brightness_ideal_low);
  }
  if (value <= config.visual_brightness_ideal_high) {
    return 1.0;
  }
  return ramp(
    config.visual_brightness_bad_high - value, 0.0,
    config.visual_brightness_bad_high - config.visual_brightness_ideal_high);
}

bool update_degraded_latch(bool current, bool valid, double quality, const QualityConfig & config)
{
  if (!valid) {
    return true;
  }
  if (current) {
    return quality < config.failure_recovery_threshold;
  }
  return quality <= config.failure_enter_threshold;
}

}  // namespace

void validate_config(const QualityConfig & config)
{
  if (config.laser_min_range_m < 0.0 ||
    config.laser_max_range_m <= config.laser_min_range_m ||
    config.laser_close_range_m <= config.laser_min_range_m ||
    config.laser_close_range_m >= config.laser_max_range_m)
  {
    throw std::invalid_argument("invalid laser range configuration");
  }
  if (config.laser_occlusion_start_ratio < 0.0 ||
    config.laser_occlusion_start_ratio >= 1.0 ||
    config.laser_occlusion_floor < 0.0 || config.laser_occlusion_floor > 1.0 ||
    config.laser_coverage_sectors <= 0 ||
    config.laser_coverage_weight < 0.0 || config.laser_coverage_weight > 1.0 ||
    config.laser_entropy_weight < 0.0 || config.laser_entropy_weight > 1.0 ||
    config.laser_geometry_weight < 0.0 || config.laser_geometry_weight > 1.0 ||
    config.laser_abrupt_jump_m <= 0.0)
  {
    throw std::invalid_argument("invalid laser scoring configuration");
  }
  if (!(config.visual_blur_bad < config.visual_blur_good) ||
    !(config.visual_contrast_bad < config.visual_contrast_good) ||
    !(config.visual_brightness_bad_low < config.visual_brightness_ideal_low) ||
    !(config.visual_brightness_ideal_low <= config.visual_brightness_ideal_high) ||
    !(config.visual_brightness_ideal_high < config.visual_brightness_bad_high))
  {
    throw std::invalid_argument("invalid visual scoring bounds");
  }
  const double visual_weight_sum =
    config.visual_blur_weight + config.visual_brightness_weight +
    config.visual_contrast_weight + config.visual_exposure_weight +
    config.visual_edge_weight;
  if (config.visual_blur_weight < 0.0 || config.visual_brightness_weight < 0.0 ||
    config.visual_contrast_weight < 0.0 || config.visual_exposure_weight < 0.0 ||
    config.visual_edge_weight < 0.0 || visual_weight_sum <= 0.0 ||
    config.visual_max_features <= 0 || config.visual_min_features < 0 ||
    config.visual_min_features > config.visual_max_features ||
    config.visual_underexposed_threshold < 0.0 ||
    config.visual_overexposed_threshold > 255.0 ||
    config.visual_underexposed_threshold >= config.visual_overexposed_threshold)
  {
    throw std::invalid_argument("invalid visual scoring weights");
  }
  if (config.failure_enter_threshold < 0.0 ||
    config.failure_recovery_threshold <= config.failure_enter_threshold ||
    config.failure_recovery_threshold > 1.0 ||
    config.failed_sensor_weight < 0.0 || config.failed_sensor_weight > 0.5 ||
    config.quality_ema_alpha <= 0.0 || config.quality_ema_alpha > 1.0 ||
    (config.fusion_strategy != "mlp" && config.fusion_strategy != "rule"))
  {
    throw std::invalid_argument("invalid fusion configuration");
  }
  if (config.visual_calibration_domain_guard_raw_min < 0.0 ||
    config.visual_calibration_domain_guard_raw_min > 1.0 ||
    config.visual_calibration_domain_guard_calibrated_max < 0.0 ||
    config.visual_calibration_domain_guard_calibrated_max > 1.0 ||
    config.visual_calibration_domain_guard_calibrated_max >=
    config.visual_calibration_domain_guard_raw_min)
  {
    throw std::invalid_argument("invalid visual calibration domain guard configuration");
  }
}

LaserQuality evaluate_laser(
  const std::vector<float> & ranges,
  double sensor_range_min,
  double sensor_range_max,
  const QualityConfig & config)
{
  const double angle_increment = ranges.empty() ?
    0.0 : 2.0 * 3.14159265358979323846 / static_cast<double>(ranges.size());
  return evaluate_laser(
    ranges, sensor_range_min, sensor_range_max,
    -3.14159265358979323846, angle_increment, config);
}

VisualQuality evaluate_visual(const cv::Mat & image, const QualityConfig & config)
{
  validate_config(config);
  VisualQuality result;
  if (image.empty() || image.rows <= 0 || image.cols <= 0) {
    result.status = "empty_image";
    return result;
  }

  try {
    cv::Mat gray;
    if (image.channels() == 1) {
      gray = image;
    } else if (image.channels() == 3) {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
      cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
      result.status = "unsupported_channels";
      return result;
    }

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar laplacian_mean;
    cv::Scalar laplacian_stddev;
    cv::meanStdDev(laplacian, laplacian_mean, laplacian_stddev);
    result.blur_variance = laplacian_stddev[0] * laplacian_stddev[0];

    cv::Scalar gray_mean;
    cv::Scalar gray_stddev;
    cv::meanStdDev(gray, gray_mean, gray_stddev);
    result.brightness = gray_mean[0];
    result.contrast = gray_stddev[0];

    result.blur_score = ramp(
      result.blur_variance, config.visual_blur_bad, config.visual_blur_good);
    result.brightness_score = brightness_score(result.brightness, config);
    result.contrast_score = ramp(
      result.contrast, config.visual_contrast_bad, config.visual_contrast_good);

    const double pixel_count = static_cast<double>(gray.total());
    result.underexposed_ratio =
      static_cast<double>(cv::countNonZero(gray < config.visual_underexposed_threshold)) /
      pixel_count;
    result.overexposed_ratio =
      static_cast<double>(cv::countNonZero(gray > config.visual_overexposed_threshold)) /
      pixel_count;
    result.illumination_level = clamp01(result.brightness / 255.0);
    result.illumination_quality = clamp01(
      result.brightness_score *
      (1.0 - result.underexposed_ratio) *
      (1.0 - result.overexposed_ratio));
    result.illumination_state = result.illumination_level < 0.25 ? "dark" :
      (result.illumination_level > 0.78 ? "strong" : "normal");

    cv::Mat edges;
    cv::Canny(gray, edges, 60.0, 120.0);
    result.edge_density =
      static_cast<double>(cv::countNonZero(edges)) / pixel_count;

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(
      gray, corners, config.visual_max_features, 0.01, 6.0);
    result.trackable_features = static_cast<int>(corners.size());
    result.motion_blur = 1.0 - result.blur_score;

    const double weight_sum =
      config.visual_blur_weight + config.visual_brightness_weight +
      config.visual_contrast_weight + config.visual_exposure_weight +
      config.visual_edge_weight;
    const double exposure_score = clamp01(
      1.0 - result.underexposed_ratio - result.overexposed_ratio);
    const double edge_score = clamp01(result.edge_density / 0.12);
    result.quality = clamp01(
      (config.visual_blur_weight * result.blur_score +
      config.visual_brightness_weight * result.brightness_score +
      config.visual_contrast_weight * result.contrast_score +
      config.visual_exposure_weight * exposure_score +
      config.visual_edge_weight * edge_score) / weight_sum);
    result.valid = true;
    result.status = result.quality <= config.failure_enter_threshold ? "degraded" : "ok";
  } catch (const cv::Exception &) {
    result = VisualQuality{};
    result.status = "opencv_error";
  }
  return result;
}

void inject_laser_occlusion(
  std::vector<float> & ranges, double severity, double close_range_m)
{
  severity = clamp01(severity);
  if (ranges.empty() || severity <= 0.0) {
    return;
  }
  const float injected_range = static_cast<float>(std::max(0.0, close_range_m));
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    // The irrational phase distributes injected samples across the full field of view.
    const double phase = std::fmod(static_cast<double>(index) * 0.61803398875, 1.0);
    if (severity >= 1.0 || phase < severity) {
      ranges[index] = injected_range;
    }
  }
}

cv::Mat inject_visual_degradation(const cv::Mat & image, double severity)
{
  severity = clamp01(severity);
  if (image.empty() || severity <= 0.0) {
    return image.clone();
  }

  cv::Mat degraded;
  int kernel = 1 + 2 * static_cast<int>(std::round(15.0 * severity));
  kernel = std::max(3, kernel);
  cv::GaussianBlur(image, degraded, cv::Size(kernel, kernel), 0.0);
  const double gain = std::max(0.0, 1.0 - 0.95 * severity);
  degraded.convertTo(degraded, -1, gain, 0.0);
  return degraded;
}

AdaptiveFusion::AdaptiveFusion(const QualityConfig & config)
: config_(config)
{
  validate_config(config_);
  if (!config_.context_model_path.empty()) {
    context_model_.load(config_.context_model_path);
  }
}

void AdaptiveFusion::reset()
{
  initialized_ = false;
  laser_degraded_ = true;
  visual_degraded_ = true;
  smoothed_laser_ = 0.0;
  smoothed_visual_ = 0.0;
}

FusionResult AdaptiveFusion::update(
  const LaserQuality & laser, const VisualQuality & visual)
{
  const double calibrated_laser_quality = laser.valid ?
    context_model_.calibrate_laser(laser) : 0.0;
  const double calibrated_visual_quality = visual.valid ?
    context_model_.calibrate_visual(visual) : 0.0;
  // A missing localization-status stream is not evidence of a lost visual
  // tracker.  If a model trained exclusively in simulation saturates low for
  // otherwise healthy image evidence, use the sensor-native scores for this
  // frame and keep the fusion model out of the hardware decision.
  const bool visual_calibration_domain_guard =
    config_.visual_calibration_domain_guard_enabled &&
    context_model_.loaded() && visual.valid && visual.tracking_ok &&
    visual.quality >= config_.visual_calibration_domain_guard_raw_min &&
    calibrated_visual_quality <=
    config_.visual_calibration_domain_guard_calibrated_max;
  const double raw_laser = visual_calibration_domain_guard ?
    clamp01(laser.quality) : calibrated_laser_quality;
  const double raw_visual = visual_calibration_domain_guard ?
    clamp01(visual.quality) : calibrated_visual_quality;
  if (!initialized_) {
    smoothed_laser_ = raw_laser;
    smoothed_visual_ = raw_visual;
    initialized_ = true;
  } else {
    const double alpha = config_.quality_ema_alpha;
    // The validity flag directly controls the current EMA state for immediate safety.
    smoothed_laser_ = laser.valid ?
      alpha * raw_laser + (1.0 - alpha) * smoothed_laser_ : 0.0;
    smoothed_visual_ = visual.valid ?
      alpha * raw_visual + (1.0 - alpha) * smoothed_visual_ : 0.0;
  }

  laser_degraded_ = update_degraded_latch(
    laser_degraded_, laser.valid, smoothed_laser_, config_);
  visual_degraded_ = update_degraded_latch(
    visual_degraded_, visual.valid, smoothed_visual_, config_);

  FusionResult result;
  result.q_laser = smoothed_laser_;
  result.q_visual = smoothed_visual_;
  result.laser_degraded = laser_degraded_;
  result.visual_degraded = visual_degraded_;
  result.model_loaded = context_model_.loaded();
  result.calibration_domain_guard_applied = visual_calibration_domain_guard;
  LaserQuality calibrated_laser = laser;
  calibrated_laser.quality = raw_laser;
  VisualQuality calibrated_visual = visual;
  calibrated_visual.quality = raw_visual;
  result.context = config_.scene_classifier_enabled ?
    context_model_.evaluate_context(calibrated_laser, calibrated_visual) :
    evaluate_environment_context(calibrated_laser, calibrated_visual);

  if (laser_degraded_ && visual_degraded_) {
    result.w_laser = 0.5;
    result.w_visual = 0.5;
    if (!laser.valid && !visual.valid) {
      result.state = "invalid_both";
    } else if (!laser.valid) {
      result.state = "invalid_laser_visual_degraded";
    } else if (!visual.valid) {
      result.state = "invalid_visual_laser_degraded";
    } else {
      result.state = "both_degraded";
    }
    result.strategy = "both_degraded";
  } else if (laser_degraded_) {
    result.w_laser = config_.failed_sensor_weight;
    result.w_visual = 1.0 - config_.failed_sensor_weight;
    result.state = laser.valid ? "laser_degraded" : "laser_invalid";
    result.strategy =
      config_.mlp_enabled && context_model_.loaded() ?
      "mlp_safety_override" : "rule_fallback";
  } else if (visual_degraded_) {
    result.w_laser = 1.0 - config_.failed_sensor_weight;
    result.w_visual = config_.failed_sensor_weight;
    result.state = visual.valid ? "visual_degraded" : "visual_invalid";
    result.strategy =
      config_.mlp_enabled && context_model_.loaded() ?
      "mlp_safety_override" : "rule_fallback";
  } else {
    bool used_model = false;
    if (!visual_calibration_domain_guard &&
      config_.mlp_enabled && config_.fusion_strategy == "mlp" &&
      context_model_.loaded())
    {
      try {
        const std::array<double, 2> predicted = context_model_.predict_weights(
          smoothed_laser_, smoothed_visual_, result.context);
        const double sum = predicted[0] + predicted[1];
        if (std::isfinite(sum) && sum > std::numeric_limits<double>::epsilon()) {
          result.w_visual = clamp01(predicted[0] / sum);
          result.w_laser = 1.0 - result.w_visual;
          result.strategy = "mlp";
          result.state = "normal";
          used_model = true;
        }
      } catch (const std::exception &) {
        used_model = false;
      }
    }
    if (!used_model) {
      const double sum = smoothed_laser_ + smoothed_visual_;
      if (sum <= std::numeric_limits<double>::epsilon()) {
        result.w_laser = 0.5;
        result.w_visual = 0.5;
      } else {
        result.w_laser = smoothed_laser_ / sum;
        result.w_visual = smoothed_visual_ / sum;
      }
      result.strategy = visual_calibration_domain_guard ?
        "domain_guard_rule_fallback" : "rule_fallback";
      result.state = visual_calibration_domain_guard ? "normal" :
        (config_.mlp_enabled && config_.fusion_strategy == "mlp" &&
        !config_.context_model_path.empty() ? "model_unavailable" : "normal");
    }
  }

  result.q_fused = clamp01(
    result.w_laser * result.q_laser + result.w_visual * result.q_visual);
  return result;
}

bool AdaptiveFusion::model_loaded() const
{
  return context_model_.loaded();
}

}  // namespace localization_quality
