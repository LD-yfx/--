#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace localization_quality
{

enum class SceneType
{
  corridor = 0,
  hall = 1,
  outdoor = 2,
  unknown = 3
};

const char * scene_type_name(SceneType scene);

struct QualityConfig
{
  double laser_min_range_m = 0.10;
  double laser_max_range_m = 10.0;
  double laser_close_range_m = 0.50;
  double laser_occlusion_start_ratio = 0.70;
  double laser_occlusion_floor = 0.10;
  int laser_coverage_sectors = 12;
  double laser_coverage_weight = 0.20;
  double laser_entropy_weight = 0.15;
  double laser_geometry_weight = 0.25;
  double laser_abrupt_jump_m = 0.50;

  double visual_blur_bad = 10.0;
  double visual_blur_good = 180.0;
  double visual_brightness_bad_low = 25.0;
  double visual_brightness_ideal_low = 70.0;
  double visual_brightness_ideal_high = 185.0;
  double visual_brightness_bad_high = 235.0;
  double visual_contrast_bad = 8.0;
  double visual_contrast_good = 45.0;
  double visual_blur_weight = 0.50;
  double visual_brightness_weight = 0.30;
  double visual_contrast_weight = 0.20;
  double visual_exposure_weight = 0.15;
  double visual_edge_weight = 0.10;
  int visual_max_features = 500;
  int visual_min_features = 40;
  double visual_underexposed_threshold = 20.0;
  double visual_overexposed_threshold = 245.0;

  double failure_enter_threshold = 0.30;
  double failure_recovery_threshold = 0.45;
  double failed_sensor_weight = 0.10;
  double quality_ema_alpha = 0.35;
  // A Gazebo-only calibrator must not turn healthy hardware imagery into a
  // failure solely because its feature distribution is out of training range.
  bool visual_calibration_domain_guard_enabled = true;
  double visual_calibration_domain_guard_raw_min = 0.35;
  double visual_calibration_domain_guard_calibrated_max = 0.10;
  bool scene_classifier_enabled = true;
  bool mlp_enabled = true;
  std::string fusion_strategy = "mlp";
  std::string context_model_path;
};

struct LaserQuality
{
  bool valid = false;
  double quality = 0.0;
  double valid_rate = 0.0;
  double close_ratio = 0.0;
  double coverage_rate = 0.0;
  double occlusion_penalty = 0.0;
  double mean_range_normalized = 0.0;
  double range_std_normalized = 0.0;
  double direction_entropy = 0.0;
  double directional_anisotropy = 0.0;
  double corridor_degeneracy = 0.0;
  double abrupt_change_rate = 0.0;
  double roughness = 0.0;
  double geometry_complexity = 0.0;
  double openness = 0.0;
  double spatial_coverage = 0.0;
  double point_density = 0.0;
  std::array<double, 3> pca_eigenvalues{{0.0, 0.0, 0.0}};
  double linearity = 0.0;
  double planarity = 0.0;
  double scattering = 0.0;
  double condition_number = 0.0;
  std::string status = "no_data";
};

struct VisualQuality
{
  bool valid = false;
  double quality = 0.0;
  double blur_variance = 0.0;
  double brightness = 0.0;
  double contrast = 0.0;
  double blur_score = 0.0;
  double brightness_score = 0.0;
  double contrast_score = 0.0;
  double underexposed_ratio = 0.0;
  double overexposed_ratio = 0.0;
  double edge_density = 0.0;
  int trackable_features = 0;
  double feature_match_rate = 0.0;
  double motion_blur = 0.0;
  bool tracking_ok = false;
  double localization_covariance = std::numeric_limits<double>::infinity();
  double illumination_level = 0.0;
  double illumination_quality = 0.0;
  std::string illumination_state = "unknown";
  std::string tracking_state = "unknown";
  std::string status = "no_data";
};

struct EnvironmentContext
{
  bool valid = false;
  SceneType scene = SceneType::unknown;
  std::array<double, 3> scene_probabilities{{0.0, 0.0, 0.0}};
  double illumination_level = 0.0;
  double illumination_quality = 0.0;
  double geometry_complexity = 0.0;
  double openness = 0.0;
  double directional_anisotropy = 0.0;
  double degeneracy = 0.0;
  double coverage_2d = 0.0;
  double coverage_3d = 0.0;
  std::string illumination_state = "unknown";
};

struct FusionResult
{
  double q_laser = 0.0;
  double q_visual = 0.0;
  double q_fused = 0.0;
  double w_laser = 0.5;
  double w_visual = 0.5;
  bool laser_degraded = true;
  bool visual_degraded = true;
  bool model_loaded = false;
  bool calibration_domain_guard_applied = false;
  std::string state = "uninitialized";
  std::string strategy = "rule";
  EnvironmentContext context;
};

void validate_config(const QualityConfig & config);

LaserQuality evaluate_laser(
  const std::vector<float> & ranges,
  double sensor_range_min,
  double sensor_range_max,
  const QualityConfig & config);

LaserQuality evaluate_laser(
  const std::vector<float> & ranges,
  double sensor_range_min,
  double sensor_range_max,
  double angle_min,
  double angle_increment,
  const QualityConfig & config);

LaserQuality evaluate_point_cloud(
  const std::vector<cv::Point3f> & points,
  const QualityConfig & config);

VisualQuality evaluate_visual(const cv::Mat & image, const QualityConfig & config);
VisualQuality evaluate_compressed_visual(
  const std::vector<std::uint8_t> & encoded,
  const QualityConfig & config);
VisualQuality apply_visual_localization_status(
  const VisualQuality & quality,
  const std::string & tracking_state,
  double covariance_trace);

class VisualQualityEvaluator
{
public:
  explicit VisualQualityEvaluator(const QualityConfig & config);
  VisualQuality evaluate(
    const cv::Mat & image,
    const std::string & tracking_state = "unknown",
    double covariance_trace = std::numeric_limits<double>::infinity());
  void reset();

private:
  QualityConfig config_;
  cv::Mat previous_descriptors_;
};

void inject_laser_occlusion(
  std::vector<float> & ranges, double severity, double close_range_m);
void inject_point_cloud_occlusion(
  std::vector<cv::Point3f> & points, double severity, double close_range_m);
cv::Mat inject_visual_degradation(const cv::Mat & image, double severity);

class ContextMlpModel
{
public:
  bool load(const std::string & path, std::string * error = nullptr);
  bool loaded() const;
  const std::string & model_path() const;
  EnvironmentContext evaluate_context(
    const LaserQuality & laser, const VisualQuality & visual) const;
  double calibrate_visual(const VisualQuality & visual) const;
  double calibrate_laser(const LaserQuality & laser) const;
  std::array<double, 2> predict_weights(
    double q_laser, double q_visual, const EnvironmentContext & context) const;

private:
  static std::vector<double> forward(
    const std::vector<double> & input,
    const cv::Mat & mean,
    const cv::Mat & stddev,
    const cv::Mat & weight1,
    const cv::Mat & bias1,
    const cv::Mat & weight2,
    const cv::Mat & bias2,
    const cv::Mat & weight3,
    const cv::Mat & bias3);

  bool loaded_ = false;
  std::string model_path_;
  cv::Mat scene_mean_, scene_std_, scene_weight1_, scene_bias1_;
  cv::Mat scene_weight2_, scene_bias2_, scene_weight3_, scene_bias3_;
  cv::Mat fusion_mean_, fusion_std_, fusion_weight1_, fusion_bias1_;
  cv::Mat fusion_weight2_, fusion_bias2_, fusion_weight3_, fusion_bias3_;
  cv::Mat visual_calibration_mean_, visual_calibration_std_;
  cv::Mat visual_calibration_weight_, visual_calibration_bias_;
  cv::Mat laser_calibration_mean_, laser_calibration_std_;
  cv::Mat laser_calibration_weight_, laser_calibration_bias_;
  bool calibration_loaded_ = false;
};

EnvironmentContext evaluate_environment_context(
  const LaserQuality & laser, const VisualQuality & visual);

double supervised_quality_label(
  double translation_error_m,
  double rotation_error_rad,
  double translation_scale_m = 0.50,
  double rotation_scale_rad = 0.35);

std::array<double, 2> oracle_weights(
  double visual_translation_error_m,
  double visual_rotation_error_rad,
  double laser_translation_error_m,
  double laser_rotation_error_rad,
  double epsilon = 1e-6);

class AdaptiveFusion
{
public:
  explicit AdaptiveFusion(const QualityConfig & config);
  FusionResult update(const LaserQuality & laser, const VisualQuality & visual);
  void reset();
  bool model_loaded() const;

private:
  QualityConfig config_;
  ContextMlpModel context_model_;
  bool initialized_ = false;
  bool laser_degraded_ = true;
  bool visual_degraded_ = true;
  double smoothed_laser_ = 0.0;
  double smoothed_visual_ = 0.0;
};

struct QualityGridConfig
{
  double resolution = 0.05;
  std::uint32_t width = 400;
  std::uint32_t height = 400;
  double origin_x = -10.0;
  double origin_y = -10.0;
};

struct QualityGridCell
{
  std::uint64_t count = 0;
  double mean = 0.0;
  double variance = 0.0;
  double confidence = 0.0;
  double last_update = 0.0;
};

class QualityGrid
{
public:
  explicit QualityGrid(const QualityGridConfig & config);
  void reset();
  bool update(double x, double y, double quality, double timestamp = 0.0);
  const QualityGridConfig & config() const;
  std::vector<std::int8_t> occupancy_data() const;
  QualityGridCell cell(std::uint32_t x, std::uint32_t y) const;
  std::uint64_t accepted_samples() const;
  std::uint64_t rejected_samples() const;

private:
  QualityGridConfig config_;
  std::vector<std::uint64_t> count_;
  std::vector<double> mean_;
  std::vector<double> m2_;
  std::vector<double> last_update_;
  std::uint64_t accepted_samples_ = 0;
  std::uint64_t rejected_samples_ = 0;
};

}  // namespace localization_quality
