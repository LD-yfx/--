#include "localization_quality/quality_core.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace localization_quality
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

double clamp01(double value)
{
  return std::max(0.0, std::min(1.0, value));
}

double safe_ratio(double numerator, double denominator)
{
  return denominator > std::numeric_limits<double>::epsilon() ?
    numerator / denominator : 0.0;
}

std::array<double, 3> softmax3(const std::array<double, 3> & logits)
{
  const double maximum = *std::max_element(logits.begin(), logits.end());
  std::array<double, 3> probabilities;
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    probabilities[i] = std::exp(logits[i] - maximum);
    sum += probabilities[i];
  }
  for (double & value : probabilities) {
    value /= sum;
  }
  return probabilities;
}

cv::Mat require_matrix(
  const cv::FileStorage & storage, const char * name, int rows = -1, int cols = -1)
{
  cv::Mat matrix;
  storage[name] >> matrix;
  if (matrix.empty()) {
    throw std::runtime_error(std::string("missing matrix: ") + name);
  }
  matrix.convertTo(matrix, CV_64F);
  if ((rows >= 0 && matrix.rows != rows) || (cols >= 0 && matrix.cols != cols)) {
    throw std::runtime_error(std::string("invalid matrix shape: ") + name);
  }
  return matrix;
}

std::vector<double> matrix_to_vector(const cv::Mat & matrix)
{
  cv::Mat flattened = matrix.reshape(1, 1);
  std::vector<double> values(static_cast<std::size_t>(flattened.cols));
  for (int column = 0; column < flattened.cols; ++column) {
    values[static_cast<std::size_t>(column)] = flattened.at<double>(0, column);
  }
  return values;
}

}  // namespace

const char * scene_type_name(SceneType scene)
{
  switch (scene) {
    case SceneType::corridor:
      return "corridor";
    case SceneType::hall:
      return "hall";
    case SceneType::outdoor:
      return "outdoor";
    default:
      return "unknown";
  }
}

LaserQuality evaluate_laser(
  const std::vector<float> & ranges,
  double sensor_range_min,
  double sensor_range_max,
  double angle_min,
  double angle_increment,
  const QualityConfig & config)
{
  validate_config(config);
  LaserQuality result;
  if (ranges.empty()) {
    result.status = "empty_scan";
    return result;
  }

  double lower = config.laser_min_range_m;
  double upper = config.laser_max_range_m;
  if (std::isfinite(sensor_range_min) && sensor_range_min >= 0.0) {
    lower = std::max(lower, sensor_range_min);
  }
  if (std::isfinite(sensor_range_max) && sensor_range_max > lower) {
    upper = std::min(upper, sensor_range_max);
  }
  if (upper <= lower || !std::isfinite(angle_min) || !std::isfinite(angle_increment)) {
    result.status = "invalid_scan_metadata";
    return result;
  }

  std::vector<cv::Point2d> points;
  points.reserve(ranges.size());
  std::vector<double> valid_ranges;
  valid_ranges.reserve(ranges.size());
  std::vector<bool> sectors(
    static_cast<std::size_t>(config.laser_coverage_sectors), false);
  std::size_t close_points = 0;
  std::size_t abrupt_changes = 0;
  double previous_range = std::numeric_limits<double>::quiet_NaN();

  for (std::size_t index = 0; index < ranges.size(); ++index) {
    const double range = static_cast<double>(ranges[index]);
    if (!std::isfinite(range) || range < lower || range > upper) {
      previous_range = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    if (std::isfinite(previous_range) &&
      std::abs(range - previous_range) >= config.laser_abrupt_jump_m)
    {
      ++abrupt_changes;
    }
    previous_range = range;
    if (range < config.laser_close_range_m) {
      ++close_points;
    }
    const double angle = angle_min + static_cast<double>(index) * angle_increment;
    points.emplace_back(range * std::cos(angle), range * std::sin(angle));
    valid_ranges.push_back(range);
    const std::size_t sector = std::min(
      sectors.size() - 1, index * sectors.size() / ranges.size());
    sectors[sector] = true;
  }

  result.valid_rate = safe_ratio(
    static_cast<double>(points.size()), static_cast<double>(ranges.size()));
  if (points.empty()) {
    result.status = "no_valid_returns";
    return result;
  }

  result.valid = true;
  result.close_ratio = safe_ratio(
    static_cast<double>(close_points), static_cast<double>(points.size()));
  result.coverage_rate = safe_ratio(
    static_cast<double>(std::count(sectors.begin(), sectors.end(), true)),
    static_cast<double>(sectors.size()));
  result.spatial_coverage = result.coverage_rate;
  result.abrupt_change_rate = safe_ratio(
    static_cast<double>(abrupt_changes),
    static_cast<double>(points.size() > 1 ? points.size() - 1 : 1));
  result.roughness = result.abrupt_change_rate;

  double range_sum = 0.0;
  for (double range : valid_ranges) {
    range_sum += range;
  }
  const double range_mean = range_sum / static_cast<double>(valid_ranges.size());
  double range_m2 = 0.0;
  for (double range : valid_ranges) {
    const double difference = range - range_mean;
    range_m2 += difference * difference;
  }
  const double range_std = std::sqrt(
    range_m2 / static_cast<double>(valid_ranges.size()));
  result.mean_range_normalized = clamp01((range_mean - lower) / (upper - lower));
  result.range_std_normalized = clamp01(range_std / (upper - lower));
  result.openness = result.mean_range_normalized;
  result.point_density = safe_ratio(
    static_cast<double>(points.size()), 2.0 * kPi * std::max(range_mean, 0.1));

  std::array<double, 12> direction_histogram{{0.0}};
  for (std::size_t index = 1; index < points.size(); ++index) {
    const double dx = points[index].x - points[index - 1].x;
    const double dy = points[index].y - points[index - 1].y;
    if (std::hypot(dx, dy) <= 1e-6) {
      continue;
    }
    double direction = std::atan2(dy, dx);
    if (direction < 0.0) {
      direction += kPi;
    }
    if (direction >= kPi) {
      direction -= kPi;
    }
    const std::size_t bin = std::min<std::size_t>(
      direction_histogram.size() - 1,
      static_cast<std::size_t>(direction / kPi * direction_histogram.size()));
    direction_histogram[bin] += 1.0;
  }
  double direction_count = 0.0;
  for (double count : direction_histogram) {
    direction_count += count;
  }
  double entropy = 0.0;
  if (direction_count > 0.0) {
    for (double count : direction_histogram) {
      if (count > 0.0) {
        const double probability = count / direction_count;
        entropy -= probability * std::log(probability);
      }
    }
    entropy /= std::log(static_cast<double>(direction_histogram.size()));
  }
  result.direction_entropy = clamp01(entropy);
  if (result.direction_entropy > 0.999) {
    result.direction_entropy = 1.0;
  }

  if (points.size() >= 3) {
    cv::Mat samples(static_cast<int>(points.size()), 2, CV_64F);
    for (int row = 0; row < samples.rows; ++row) {
      samples.at<double>(row, 0) = points[static_cast<std::size_t>(row)].x;
      samples.at<double>(row, 1) = points[static_cast<std::size_t>(row)].y;
    }
    cv::PCA pca(samples, cv::Mat(), cv::PCA::DATA_AS_ROW);
    const double eigen0 = std::max(0.0, pca.eigenvalues.at<double>(0, 0));
    const double eigen1 = std::max(0.0, pca.eigenvalues.at<double>(1, 0));
    result.pca_eigenvalues = {{eigen0, eigen1, 0.0}};
    result.directional_anisotropy = clamp01(1.0 - safe_ratio(eigen1, eigen0));
    result.linearity = result.directional_anisotropy;
    result.planarity = 1.0 - result.linearity;
    result.condition_number = eigen0 / std::max(eigen1, 1e-9);
  }
  result.corridor_degeneracy = clamp01(
    0.65 * result.directional_anisotropy +
    0.35 * (1.0 - result.direction_entropy));
  result.geometry_complexity = clamp01(
    0.45 * result.direction_entropy +
    0.30 * (1.0 - result.directional_anisotropy) +
    0.25 * std::min(1.0, result.abrupt_change_rate / 0.20));

  result.occlusion_penalty = 1.0;
  if (result.close_ratio > config.laser_occlusion_start_ratio) {
    const double progress =
      (result.close_ratio - config.laser_occlusion_start_ratio) /
      (1.0 - config.laser_occlusion_start_ratio);
    result.occlusion_penalty = std::max(
      config.laser_occlusion_floor,
      1.0 - progress * (1.0 - config.laser_occlusion_floor));
  }

  const double fixed_weight = std::max(
    0.0, 1.0 - config.laser_coverage_weight -
    config.laser_entropy_weight - config.laser_geometry_weight);
  const double evidence =
    fixed_weight +
    config.laser_coverage_weight * result.coverage_rate +
    config.laser_entropy_weight * result.direction_entropy +
    config.laser_geometry_weight * (1.0 - 0.70 * result.corridor_degeneracy);
  result.quality = clamp01(result.valid_rate * result.occlusion_penalty * evidence);
  result.status = result.close_ratio > config.laser_occlusion_start_ratio ? "occluded" :
    (result.quality <= config.failure_enter_threshold ? "degraded_geometry" : "ok");
  return result;
}

LaserQuality evaluate_point_cloud(
  const std::vector<cv::Point3f> & points, const QualityConfig & config)
{
  validate_config(config);
  LaserQuality result;
  if (points.empty()) {
    result.status = "empty_cloud";
    return result;
  }

  std::vector<cv::Point3d> valid;
  valid.reserve(points.size());
  std::size_t close_points = 0;
  for (const cv::Point3f & point : points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    const double range = std::sqrt(
      static_cast<double>(point.x) * point.x +
      static_cast<double>(point.y) * point.y +
      static_cast<double>(point.z) * point.z);
    if (range < config.laser_min_range_m || range > config.laser_max_range_m) {
      continue;
    }
    if (range < config.laser_close_range_m) {
      ++close_points;
    }
    valid.emplace_back(point.x, point.y, point.z);
  }
  result.valid_rate = safe_ratio(
    static_cast<double>(valid.size()), static_cast<double>(points.size()));
  if (valid.empty()) {
    result.status = "no_valid_points";
    return result;
  }
  if (valid.size() < 3) {
    result.status = "insufficient_points";
    return result;
  }
  result.valid = true;
  result.close_ratio = safe_ratio(
    static_cast<double>(close_points), static_cast<double>(valid.size()));

  cv::Mat samples(static_cast<int>(valid.size()), 3, CV_64F);
  std::array<double, 3> minimum{{
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()}};
  std::array<double, 3> maximum{{
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity()}};
  double mean_range = 0.0;
  for (int row = 0; row < samples.rows; ++row) {
    const cv::Point3d & point = valid[static_cast<std::size_t>(row)];
    samples.at<double>(row, 0) = point.x;
    samples.at<double>(row, 1) = point.y;
    samples.at<double>(row, 2) = point.z;
    const std::array<double, 3> coordinates{{point.x, point.y, point.z}};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      minimum[axis] = std::min(minimum[axis], coordinates[axis]);
      maximum[axis] = std::max(maximum[axis], coordinates[axis]);
    }
    mean_range += std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
  }
  mean_range /= static_cast<double>(valid.size());
  result.mean_range_normalized = clamp01(
    (mean_range - config.laser_min_range_m) /
    (config.laser_max_range_m - config.laser_min_range_m));
  result.openness = result.mean_range_normalized;

  cv::PCA pca(samples, cv::Mat(), cv::PCA::DATA_AS_ROW);
  const double eigen0 = std::max(0.0, pca.eigenvalues.at<double>(0, 0));
  const double eigen1 = std::max(0.0, pca.eigenvalues.at<double>(1, 0));
  const double eigen2 = std::max(0.0, pca.eigenvalues.at<double>(2, 0));
  result.pca_eigenvalues = {{eigen0, eigen1, eigen2}};
  if (eigen0 > 1e-12) {
    result.linearity = clamp01((eigen0 - eigen1) / eigen0);
    result.planarity = clamp01((eigen1 - eigen2) / eigen0);
    result.scattering = clamp01(eigen2 / eigen0);
  }
  result.directional_anisotropy = result.linearity;
  result.condition_number = eigen0 / std::max(eigen2, 1e-9);
  const double degeneracy = clamp01(
    std::max(result.linearity, result.planarity) * (1.0 - result.scattering));
  result.corridor_degeneracy = degeneracy;
  result.geometry_complexity = clamp01(
    0.45 * result.scattering + 0.35 * result.planarity +
    0.20 * (1.0 - result.linearity));

  std::array<bool, 64> occupied{{false}};
  for (const cv::Point3d & point : valid) {
    const std::array<double, 3> coordinates{{point.x, point.y, point.z}};
    std::array<std::size_t, 3> bin{{0, 0, 0}};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const double extent = maximum[axis] - minimum[axis];
      const double normalized = extent > 1e-9 ?
        (coordinates[axis] - minimum[axis]) / extent : 0.0;
      bin[axis] = std::min<std::size_t>(3, static_cast<std::size_t>(normalized * 4.0));
    }
    occupied[bin[0] + 4 * bin[1] + 16 * bin[2]] = true;
  }
  result.spatial_coverage = safe_ratio(
    static_cast<double>(std::count(occupied.begin(), occupied.end(), true)), 64.0);
  result.coverage_rate = result.spatial_coverage;
  const double bounding_volume =
    std::max(1e-6, (maximum[0] - minimum[0]) *
    (maximum[1] - minimum[1]) * (maximum[2] - minimum[2]));
  result.point_density = static_cast<double>(valid.size()) / bounding_volume;

  result.occlusion_penalty = result.close_ratio <= config.laser_occlusion_start_ratio ?
    1.0 : std::max(
    config.laser_occlusion_floor,
    1.0 - (result.close_ratio - config.laser_occlusion_start_ratio) /
    (1.0 - config.laser_occlusion_start_ratio));
  const double geometry_evidence = clamp01(
    0.35 + 0.35 * result.spatial_coverage +
    0.30 * (1.0 - degeneracy));
  result.quality = clamp01(
    result.valid_rate * result.occlusion_penalty * geometry_evidence);
  result.status = result.close_ratio > config.laser_occlusion_start_ratio ? "occluded" :
    (result.quality <= config.failure_enter_threshold ? "degraded_geometry" : "ok");
  return result;
}

void inject_point_cloud_occlusion(
  std::vector<cv::Point3f> & points, double severity, double close_range_m)
{
  severity = clamp01(severity);
  if (severity <= 0.0) {
    return;
  }
  for (std::size_t index = 0; index < points.size(); ++index) {
    const double phase = std::fmod(static_cast<double>(index) * 0.61803398875, 1.0);
    if (severity >= 1.0 || phase < severity) {
      const double angle = 2.0 * kPi * phase;
      points[index] = cv::Point3f(
        static_cast<float>(close_range_m * std::cos(angle)),
        static_cast<float>(close_range_m * std::sin(angle)), 0.0F);
    }
  }
}

VisualQuality evaluate_compressed_visual(
  const std::vector<std::uint8_t> & encoded,
  const QualityConfig & config)
{
  if (encoded.empty()) {
    VisualQuality result;
    result.status = "empty_compressed_image";
    return result;
  }
  try {
    const cv::Mat buffer(
      1, static_cast<int>(encoded.size()), CV_8UC1,
      const_cast<std::uint8_t *>(encoded.data()));
    const cv::Mat image = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      VisualQuality result;
      result.status = "compressed_decode_error";
      return result;
    }
    return evaluate_visual(image, config);
  } catch (const cv::Exception &) {
    VisualQuality result;
    result.status = "compressed_decode_error";
    return result;
  }
}

VisualQuality apply_visual_localization_status(
  const VisualQuality & quality,
  const std::string & tracking_state,
  double covariance_trace)
{
  VisualQuality result = quality;
  result.tracking_state = tracking_state;
  result.localization_covariance = covariance_trace;
  const bool state_available = tracking_state != "unknown" && !tracking_state.empty();
  result.tracking_ok = !state_available ||
    tracking_state == "ok" || tracking_state == "tracking";
  double covariance_score = 1.0;
  if (std::isfinite(covariance_trace)) {
    covariance_score = std::exp(-std::max(0.0, covariance_trace) / 0.25);
  }
  if (state_available) {
    result.quality = clamp01(
      result.quality * (result.tracking_ok ? (0.7 + 0.3 * covariance_score) : 0.15));
  }
  if (!result.tracking_ok) {
    result.status = "tracking_lost";
  }
  return result;
}

VisualQualityEvaluator::VisualQualityEvaluator(const QualityConfig & config)
: config_(config)
{
  validate_config(config_);
}

VisualQuality VisualQualityEvaluator::evaluate(
  const cv::Mat & image,
  const std::string & tracking_state,
  double covariance_trace)
{
  VisualQuality result = evaluate_visual(image, config_);
  if (!result.valid) {
    previous_descriptors_.release();
    return result;
  }
  try {
    cv::Mat gray;
    if (image.channels() == 1) {
      gray = image;
    } else if (image.channels() == 3) {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
      cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config_.visual_max_features);
    orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    result.trackable_features = static_cast<int>(keypoints.size());
    const bool had_previous =
      !previous_descriptors_.empty() && !descriptors.empty();
    if (had_previous) {
      std::vector<std::vector<cv::DMatch>> matches;
      cv::BFMatcher(cv::NORM_HAMMING).knnMatch(
        previous_descriptors_, descriptors, matches, 2);
      std::size_t good_matches = 0;
      for (const auto & pair : matches) {
        if (pair.size() >= 2 && pair[0].distance < 0.75F * pair[1].distance) {
          ++good_matches;
        }
      }
      result.feature_match_rate = safe_ratio(
        static_cast<double>(good_matches),
        static_cast<double>(std::min(
          previous_descriptors_.rows, descriptors.rows)));
    }
    previous_descriptors_ = descriptors.clone();
    const double feature_score = clamp01(
      static_cast<double>(result.trackable_features) /
      std::max(1, config_.visual_min_features));
    const double temporal_score = !had_previous ?
      feature_score : 0.6 * feature_score + 0.4 * result.feature_match_rate;
    result.quality = clamp01(0.85 * result.quality + 0.15 * temporal_score);
    result = apply_visual_localization_status(result, tracking_state, covariance_trace);
  } catch (const cv::Exception &) {
    previous_descriptors_.release();
    result.valid = false;
    result.quality = 0.0;
    result.status = "feature_error";
  }
  return result;
}

void VisualQualityEvaluator::reset()
{
  previous_descriptors_.release();
}

EnvironmentContext evaluate_environment_context(
  const LaserQuality & laser, const VisualQuality & visual)
{
  EnvironmentContext context;
  context.valid = laser.valid || visual.valid;
  context.illumination_level = visual.illumination_level;
  context.illumination_quality = visual.illumination_quality;
  context.illumination_state = visual.illumination_state;
  context.geometry_complexity = laser.geometry_complexity;
  context.openness = laser.openness;
  context.directional_anisotropy = laser.directional_anisotropy;
  context.degeneracy = laser.corridor_degeneracy;
  context.coverage_2d = laser.coverage_rate;
  context.coverage_3d = laser.scattering > 0.0 ? laser.spatial_coverage : 0.0;
  if (!context.valid) {
    return context;
  }
  const std::array<double, 3> logits{{
    2.8 * laser.corridor_degeneracy + 0.8 * (1.0 - laser.openness),
    1.5 * laser.openness + 1.2 * laser.geometry_complexity -
      0.6 * laser.corridor_degeneracy,
    2.4 * laser.openness + 0.5 * visual.illumination_level -
      0.8 * laser.geometry_complexity}};
  context.scene_probabilities = softmax3(logits);
  const auto best = std::max_element(
    context.scene_probabilities.begin(), context.scene_probabilities.end());
  context.scene = static_cast<SceneType>(
    std::distance(context.scene_probabilities.begin(), best));
  return context;
}

bool ContextMlpModel::load(const std::string & path, std::string * error)
{
  loaded_ = false;
  calibration_loaded_ = false;
  model_path_.clear();
  try {
    cv::FileStorage storage(path, cv::FileStorage::READ);
    if (!storage.isOpened()) {
      throw std::runtime_error("cannot open model file");
    }
    int format_version = 0;
    storage["format_version"] >> format_version;
    if (format_version != 1) {
      throw std::runtime_error("unsupported model format_version");
    }
    scene_mean_ = require_matrix(storage, "scene_mean", 1);
    scene_std_ = require_matrix(storage, "scene_std", 1, scene_mean_.cols);
    scene_weight1_ = require_matrix(storage, "scene_weight1", -1, scene_mean_.cols);
    scene_bias1_ = require_matrix(storage, "scene_bias1", scene_weight1_.rows, 1);
    scene_weight2_ = require_matrix(storage, "scene_weight2", -1, scene_weight1_.rows);
    scene_bias2_ = require_matrix(storage, "scene_bias2", scene_weight2_.rows, 1);
    scene_weight3_ = require_matrix(storage, "scene_weight3", 3, scene_weight2_.rows);
    scene_bias3_ = require_matrix(storage, "scene_bias3", 3, 1);

    fusion_mean_ = require_matrix(storage, "fusion_mean", 1);
    fusion_std_ = require_matrix(storage, "fusion_std", 1, fusion_mean_.cols);
    fusion_weight1_ = require_matrix(storage, "fusion_weight1", -1, fusion_mean_.cols);
    fusion_bias1_ = require_matrix(storage, "fusion_bias1", fusion_weight1_.rows, 1);
    fusion_weight2_ = require_matrix(storage, "fusion_weight2", -1, fusion_weight1_.rows);
    fusion_bias2_ = require_matrix(storage, "fusion_bias2", fusion_weight2_.rows, 1);
    fusion_weight3_ = require_matrix(storage, "fusion_weight3", 2, fusion_weight2_.rows);
    fusion_bias3_ = require_matrix(storage, "fusion_bias3", 2, 1);
    calibration_loaded_ = false;
    if (!storage["visual_calibration_mean"].empty() &&
      !storage["laser_calibration_mean"].empty())
    {
      visual_calibration_mean_ = require_matrix(
        storage, "visual_calibration_mean", 1);
      visual_calibration_std_ = require_matrix(
        storage, "visual_calibration_std", 1, visual_calibration_mean_.cols);
      visual_calibration_weight_ = require_matrix(
        storage, "visual_calibration_weight", 1, visual_calibration_mean_.cols);
      visual_calibration_bias_ = require_matrix(
        storage, "visual_calibration_bias", 1, 1);
      laser_calibration_mean_ = require_matrix(
        storage, "laser_calibration_mean", 1);
      laser_calibration_std_ = require_matrix(
        storage, "laser_calibration_std", 1, laser_calibration_mean_.cols);
      laser_calibration_weight_ = require_matrix(
        storage, "laser_calibration_weight", 1, laser_calibration_mean_.cols);
      laser_calibration_bias_ = require_matrix(
        storage, "laser_calibration_bias", 1, 1);
      calibration_loaded_ = true;
    }
    loaded_ = true;
    model_path_ = path;
    return true;
  } catch (const std::exception & exception) {
    if (error != nullptr) {
      *error = exception.what();
    }
    return false;
  }
}

bool ContextMlpModel::loaded() const
{
  return loaded_;
}

const std::string & ContextMlpModel::model_path() const
{
  return model_path_;
}

double ContextMlpModel::calibrate_visual(const VisualQuality & visual) const
{
  if (!calibration_loaded_) {
    return clamp01(visual.quality);
  }
  std::vector<double> features{
    visual.quality, visual.illumination_quality,
    visual.underexposed_ratio, visual.overexposed_ratio,
    visual.edge_density, safe_ratio(visual.trackable_features, 500.0),
    visual.feature_match_rate};
  if (visual_calibration_mean_.cols >= 9) {
    const double covariance_score = std::isfinite(visual.localization_covariance) ?
      std::exp(-std::max(0.0, visual.localization_covariance) / 0.25) : 0.0;
    features.push_back(visual.tracking_ok ? 1.0 : 0.0);
    features.push_back(covariance_score);
  }
  if (static_cast<int>(features.size()) != visual_calibration_mean_.cols) {
    throw std::runtime_error("visual calibration input dimension mismatch");
  }
  double logit = visual_calibration_bias_.at<double>(0, 0);
  for (std::size_t index = 0; index < features.size(); ++index) {
    const double scale =
      std::abs(visual_calibration_std_.at<double>(0, static_cast<int>(index))) <
      1e-12 ? 1.0 :
      visual_calibration_std_.at<double>(0, static_cast<int>(index));
    logit += visual_calibration_weight_.at<double>(
      0, static_cast<int>(index)) *
      (features[index] - visual_calibration_mean_.at<double>(
      0, static_cast<int>(index))) / scale;
  }
  return 1.0 / (1.0 + std::exp(-std::max(-40.0, std::min(40.0, logit))));
}

double ContextMlpModel::calibrate_laser(const LaserQuality & laser) const
{
  if (!calibration_loaded_) {
    return clamp01(laser.quality);
  }
  const std::vector<double> features{
    laser.quality, laser.valid_rate, laser.close_ratio,
    laser.coverage_rate, laser.direction_entropy,
    laser.directional_anisotropy, laser.geometry_complexity,
    laser.openness, laser.linearity};
  double logit = laser_calibration_bias_.at<double>(0, 0);
  for (std::size_t index = 0; index < features.size(); ++index) {
    const double scale =
      std::abs(laser_calibration_std_.at<double>(0, static_cast<int>(index))) <
      1e-12 ? 1.0 :
      laser_calibration_std_.at<double>(0, static_cast<int>(index));
    logit += laser_calibration_weight_.at<double>(
      0, static_cast<int>(index)) *
      (features[index] - laser_calibration_mean_.at<double>(
      0, static_cast<int>(index))) / scale;
  }
  return 1.0 / (1.0 + std::exp(-std::max(-40.0, std::min(40.0, logit))));
}

std::vector<double> ContextMlpModel::forward(
  const std::vector<double> & input,
  const cv::Mat & mean,
  const cv::Mat & stddev,
  const cv::Mat & weight1,
  const cv::Mat & bias1,
  const cv::Mat & weight2,
  const cv::Mat & bias2,
  const cv::Mat & weight3,
  const cv::Mat & bias3)
{
  if (static_cast<int>(input.size()) != mean.cols) {
    throw std::invalid_argument("model input dimension mismatch");
  }
  cv::Mat normalized(mean.cols, 1, CV_64F);
  for (int row = 0; row < mean.cols; ++row) {
    const double scale = std::abs(stddev.at<double>(0, row)) < 1e-12 ?
      1.0 : stddev.at<double>(0, row);
    normalized.at<double>(row, 0) =
      (input[static_cast<std::size_t>(row)] - mean.at<double>(0, row)) / scale;
  }
  cv::Mat hidden1 = weight1 * normalized + bias1;
  cv::max(hidden1, 0.0, hidden1);
  cv::Mat hidden2 = weight2 * hidden1 + bias2;
  cv::max(hidden2, 0.0, hidden2);
  cv::Mat logits = weight3 * hidden2 + bias3;
  const double maximum = *std::max_element(
    logits.begin<double>(), logits.end<double>());
  double sum = 0.0;
  for (int row = 0; row < logits.rows; ++row) {
    logits.at<double>(row, 0) = std::exp(logits.at<double>(row, 0) - maximum);
    sum += logits.at<double>(row, 0);
  }
  logits /= sum;
  return matrix_to_vector(logits);
}

EnvironmentContext ContextMlpModel::evaluate_context(
  const LaserQuality & laser, const VisualQuality & visual) const
{
  EnvironmentContext context = evaluate_environment_context(laser, visual);
  if (!loaded_) {
    return context;
  }
  std::vector<double> input;
  if (scene_mean_.cols == 11) {
    input = {
      laser.openness, laser.direction_entropy,
      laser.directional_anisotropy, laser.geometry_complexity,
      laser.coverage_rate, laser.linearity, laser.planarity,
      laser.valid_rate, laser.close_ratio, laser.abrupt_change_rate,
      laser.corridor_degeneracy};
  } else {
    // Backward-compatible v0.2/bootstrap schema.
    input = {
      visual.illumination_level, visual.illumination_quality,
      visual.underexposed_ratio, visual.overexposed_ratio,
      visual.edge_density,
      safe_ratio(visual.trackable_features, 500.0),
      visual.feature_match_rate, laser.openness, laser.direction_entropy,
      laser.directional_anisotropy, laser.geometry_complexity,
      laser.coverage_rate, laser.linearity, laser.planarity};
  }
  const std::vector<double> probabilities = forward(
    input, scene_mean_, scene_std_, scene_weight1_, scene_bias1_,
    scene_weight2_, scene_bias2_, scene_weight3_, scene_bias3_);
  context.scene_probabilities = {{
    probabilities[0], probabilities[1], probabilities[2]}};
  context.scene = static_cast<SceneType>(std::distance(
    probabilities.begin(),
    std::max_element(probabilities.begin(), probabilities.end())));
  return context;
}

std::array<double, 2> ContextMlpModel::predict_weights(
  double q_laser, double q_visual, const EnvironmentContext & context) const
{
  if (!loaded_) {
    throw std::runtime_error("model is not loaded");
  }
  const std::vector<double> input{
    q_visual, q_laser,
    context.scene_probabilities[0], context.scene_probabilities[1],
    context.scene_probabilities[2], context.illumination_quality,
    context.geometry_complexity, context.openness,
    q_visual > 0.0 ? 1.0 : 0.0, q_laser > 0.0 ? 1.0 : 0.0};
  const std::vector<double> output = forward(
    input, fusion_mean_, fusion_std_, fusion_weight1_, fusion_bias1_,
    fusion_weight2_, fusion_bias2_, fusion_weight3_, fusion_bias3_);
  return {{output[0], output[1]}};
}

double supervised_quality_label(
  double translation_error_m,
  double rotation_error_rad,
  double translation_scale_m,
  double rotation_scale_rad)
{
  if (!std::isfinite(translation_error_m) || !std::isfinite(rotation_error_rad) ||
    translation_scale_m <= 0.0 || rotation_scale_rad <= 0.0)
  {
    return 0.0;
  }
  const double normalized_translation =
    std::max(0.0, translation_error_m) / translation_scale_m;
  const double normalized_rotation =
    std::max(0.0, rotation_error_rad) / rotation_scale_rad;
  return clamp01(std::exp(
    -std::sqrt(normalized_translation * normalized_translation +
    normalized_rotation * normalized_rotation)));
}

std::array<double, 2> oracle_weights(
  double visual_translation_error_m,
  double visual_rotation_error_rad,
  double laser_translation_error_m,
  double laser_rotation_error_rad,
  double epsilon)
{
  const double visual_loss =
    visual_translation_error_m * visual_translation_error_m +
    visual_rotation_error_rad * visual_rotation_error_rad;
  const double laser_loss =
    laser_translation_error_m * laser_translation_error_m +
    laser_rotation_error_rad * laser_rotation_error_rad;
  if (!std::isfinite(visual_loss) && !std::isfinite(laser_loss)) {
    return {{0.5, 0.5}};
  }
  if (!std::isfinite(visual_loss)) {
    return {{0.0, 1.0}};
  }
  if (!std::isfinite(laser_loss)) {
    return {{1.0, 0.0}};
  }
  const double visual_reliability = 1.0 / (std::max(0.0, visual_loss) + epsilon);
  const double laser_reliability = 1.0 / (std::max(0.0, laser_loss) + epsilon);
  const double sum = visual_reliability + laser_reliability;
  return {{visual_reliability / sum, laser_reliability / sum}};
}

QualityGrid::QualityGrid(const QualityGridConfig & config)
: config_(config)
{
  if (config_.resolution <= 0.0 || config_.width == 0 || config_.height == 0) {
    throw std::invalid_argument("invalid quality grid configuration");
  }
  reset();
}

void QualityGrid::reset()
{
  const std::size_t size =
    static_cast<std::size_t>(config_.width) * config_.height;
  count_.assign(size, 0);
  mean_.assign(size, 0.0);
  m2_.assign(size, 0.0);
  last_update_.assign(size, 0.0);
  accepted_samples_ = 0;
  rejected_samples_ = 0;
}

bool QualityGrid::update(
  double x, double y, double quality, double timestamp)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(quality) ||
    quality < 0.0 || quality > 1.0)
  {
    ++rejected_samples_;
    return false;
  }
  const long grid_x = static_cast<long>(
    std::floor((x - config_.origin_x) / config_.resolution));
  const long grid_y = static_cast<long>(
    std::floor((y - config_.origin_y) / config_.resolution));
  if (grid_x < 0 || grid_y < 0 ||
    grid_x >= static_cast<long>(config_.width) ||
    grid_y >= static_cast<long>(config_.height))
  {
    ++rejected_samples_;
    return false;
  }
  const std::size_t index =
    static_cast<std::size_t>(grid_y) * config_.width +
    static_cast<std::size_t>(grid_x);
  ++count_[index];
  const double difference = quality - mean_[index];
  mean_[index] += difference / static_cast<double>(count_[index]);
  const double difference2 = quality - mean_[index];
  m2_[index] += difference * difference2;
  last_update_[index] = timestamp;
  ++accepted_samples_;
  return true;
}

const QualityGridConfig & QualityGrid::config() const
{
  return config_;
}

std::vector<std::int8_t> QualityGrid::occupancy_data() const
{
  std::vector<std::int8_t> data(count_.size(), static_cast<std::int8_t>(-1));
  for (std::size_t index = 0; index < count_.size(); ++index) {
    if (count_[index] > 0) {
      data[index] = static_cast<std::int8_t>(
        std::lround(clamp01(mean_[index]) * 100.0));
    }
  }
  return data;
}

QualityGridCell QualityGrid::cell(std::uint32_t x, std::uint32_t y) const
{
  if (x >= config_.width || y >= config_.height) {
    throw std::out_of_range("quality grid cell index out of range");
  }
  const std::size_t index = static_cast<std::size_t>(y) * config_.width + x;
  QualityGridCell result;
  result.count = count_[index];
  result.mean = mean_[index];
  result.variance = count_[index] > 1 ?
    m2_[index] / static_cast<double>(count_[index] - 1) : 0.0;
  result.confidence = 1.0 - std::exp(-static_cast<double>(count_[index]) / 5.0);
  result.last_update = last_update_[index];
  return result;
}

std::uint64_t QualityGrid::accepted_samples() const
{
  return accepted_samples_;
}

std::uint64_t QualityGrid::rejected_samples() const
{
  return rejected_samples_;
}

}  // namespace localization_quality
