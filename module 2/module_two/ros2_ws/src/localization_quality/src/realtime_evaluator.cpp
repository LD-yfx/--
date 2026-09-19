#include "localization_quality/causal_sync.hpp"
#include "localization_quality/quality_core.hpp"
#include "localization_quality/quality_statistics.hpp"
#include "localization_quality/ros_parameters.hpp"

#include "cv_bridge/cv_bridge.h"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace lq = localization_quality;
using std::placeholders::_1;

namespace
{

template<typename MessageT>
double message_time_seconds(const MessageT & message, double fallback)
{
  const double timestamp = static_cast<double>(message.header.stamp.sec) +
    static_cast<double>(message.header.stamp.nanosec) / 1e9;
  return timestamp > 0.0 ? timestamp : fallback;
}

diagnostic_msgs::msg::KeyValue key_value(
  const std::string & key, const std::string & value)
{
  diagnostic_msgs::msg::KeyValue item;
  item.key = key;
  item.value = value;
  return item;
}

std::string number(double value)
{
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream stream;
  stream << std::setprecision(9) << value;
  return stream.str();
}

double covariance_trace(const nav_msgs::msg::Odometry & message)
{
  const auto & covariance = message.pose.covariance;
  double trace = 0.0;
  for (std::size_t index : {0U, 7U, 14U, 21U, 28U, 35U}) {
    if (!std::isfinite(covariance[index]) || covariance[index] < 0.0) {
      return std::numeric_limits<double>::infinity();
    }
    trace += covariance[index];
  }
  return trace;
}

rclcpp::QoS configured_qos(
  std::size_t depth,
  const std::string & reliability,
  const std::string & durability,
  const std::string & parameter_prefix)
{
  rclcpp::QoS qos{rclcpp::KeepLast(depth)};
  if (reliability == "reliable") {
    qos.reliable();
  } else if (reliability == "best_effort") {
    qos.best_effort();
  } else {
    throw std::invalid_argument(
            parameter_prefix + ".reliability must be reliable or best_effort");
  }
  if (durability == "transient_local") {
    qos.transient_local();
  } else if (durability == "volatile") {
    qos.durability_volatile();
  } else {
    throw std::invalid_argument(
            parameter_prefix +
            ".durability must be volatile or transient_local");
  }
  return qos;
}

struct QosSettings
{
  std::string reliability;
  std::string durability;
  int depth = 20;
};

QosSettings declare_qos(
  rclcpp::Node & node,
  const std::string & prefix,
  const std::string & default_reliability)
{
  QosSettings settings;
  settings.reliability = node.declare_parameter<std::string>(
    prefix + ".reliability", default_reliability);
  settings.durability = node.declare_parameter<std::string>(
    prefix + ".durability", "volatile");
  settings.depth = node.declare_parameter<int>(prefix + ".depth", 20);
  if (settings.depth <= 0) {
    throw std::invalid_argument(prefix + ".depth must be positive");
  }
  return settings;
}

struct PoseSample
{
  double x = 0.0;
  double y = 0.0;
  std::string frame_id;
};

}  // namespace

class RealtimeEvaluator : public rclcpp::Node
{
public:
  RealtimeEvaluator()
  : Node("realtime_evaluator")
  {
    quality_config_ = lq::declare_quality_parameters(*this);
    visual_evaluator_ = std::make_unique<lq::VisualQualityEvaluator>(quality_config_);
    fusion_ = std::make_unique<lq::AdaptiveFusion>(quality_config_);

    scan_topic_ = declare_parameter<std::string>("topics.scan", "/scan");
    point_cloud_topic_ = declare_parameter<std::string>(
      "topics.point_cloud", "/points");
    image_topic_ = declare_parameter<std::string>(
      "topics.image", "/camera/image_raw");
    compressed_image_topic_ = declare_parameter<std::string>(
      "topics.compressed_image", "/camera/image_raw/compressed");
    odom_topic_ = declare_parameter<std::string>("topics.odom", "/odom");
    visual_localization_topic_ = declare_parameter<std::string>(
      "topics.visual_localization", "/visual_localization/odom");
    laser_localization_topic_ = declare_parameter<std::string>(
      "topics.laser_localization", "/laser_localization/odom");
    metrics_topic_ = declare_parameter<std::string>(
      "topics.metrics", "/localization_quality/metrics");
    diagnostics_topic_ = declare_parameter<std::string>(
      "topics.diagnostics", "/localization_quality/diagnostics");
    quality_map_topic_ = declare_parameter<std::string>(
      "topics.quality_map", "/localization_quality/quality_map");
    quality_stats_topic_ = declare_parameter<std::string>(
      "topics.quality_stats", "/localization_quality/quality_stats");
    publish_statistics_ = declare_parameter<bool>("map.publish_statistics", true);
    statistics_publish_period_sec_ = declare_parameter<double>(
      "map.statistics_publish_period_sec", 0.5);
    const double statistics_window_sec = declare_parameter<double>(
      "map.statistics_window_sec", 10.0);
    const int statistics_max_samples = declare_parameter<int>(
      "map.statistics_max_samples_per_cell", 256);
    if (!std::isfinite(statistics_publish_period_sec_) ||
      statistics_publish_period_sec_ <= 0.0 || statistics_max_samples <= 0)
    {
      throw std::invalid_argument("statistics publish period and per-cell capacity must be positive");
    }

    image_transport_ = declare_parameter<std::string>("input.image", "raw");
    laser_input_ = declare_parameter<std::string>("input.laser", "scan");
    odom_required_ = declare_parameter<bool>("input.odom_required", true);
    sync_tolerance_sec_ = declare_parameter<double>("sync.tolerance_sec", 0.10);
    sensor_timeout_sec_ = declare_parameter<double>("sync.sensor_timeout_sec", 0.50);
    queue_depth_ = declare_parameter<int>("sync.queue_depth", 50);
    target_frame_ = declare_parameter<std::string>("frames.target", "odom");
    tracking_covariance_limit_ = declare_parameter<double>(
      "quality.visual.tracking_covariance_limit", 0.50);
    output_csv_ = declare_parameter<std::string>("output_csv", "");
    run_mode_ = declare_parameter<std::string>("run_mode", "normal");
    fault_severity_ = declare_parameter<double>("fault_severity", 1.0);

    image_qos_ = declare_qos(*this, "qos.image", "best_effort");
    laser_qos_ = declare_qos(*this, "qos.laser", "best_effort");
    odom_qos_ = declare_qos(*this, "qos.odom", "best_effort");
    localization_qos_ = declare_qos(
      *this, "qos.localization", "reliable");

    lq::QualityGridConfig grid_config;
    grid_config.resolution = declare_parameter<double>(
      "map.resolution", grid_config.resolution);
    const int map_width = declare_parameter<int>(
      "map.width", static_cast<int>(grid_config.width));
    const int map_height = declare_parameter<int>(
      "map.height", static_cast<int>(grid_config.height));
    if (map_width <= 0 || map_height <= 0 ||
      static_cast<std::uint64_t>(map_width) * static_cast<std::uint64_t>(map_height) >
      std::numeric_limits<std::uint32_t>::max())
    {
      throw std::invalid_argument("map dimensions must be positive with uint32 cell indices");
    }
    grid_config.width = static_cast<std::uint32_t>(map_width);
    grid_config.height = static_cast<std::uint32_t>(map_height);
    grid_config.origin_x = declare_parameter<double>(
      "map.origin_x", grid_config.origin_x);
    grid_config.origin_y = declare_parameter<double>(
      "map.origin_y", grid_config.origin_y);
    if (!std::isfinite(grid_config.resolution) || grid_config.resolution <= 0.0 ||
      !std::isfinite(grid_config.origin_x) || !std::isfinite(grid_config.origin_y))
    {
      throw std::invalid_argument("map geometry must be finite with positive resolution");
    }
    grid_ = std::make_unique<lq::QualityGrid>(grid_config);
    if (publish_statistics_) {
      statistics_ = std::make_unique<lq::QualityStatisticsAccumulator>(
        grid_config, statistics_window_sec, static_cast<std::size_t>(statistics_max_samples));
    }
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    if (image_transport_ != "raw" && image_transport_ != "compressed") {
      throw std::invalid_argument("input.image must be raw or compressed");
    }
    if (laser_input_ != "scan" && laser_input_ != "pointcloud2") {
      throw std::invalid_argument("input.laser must be scan or pointcloud2");
    }
    if (queue_depth_ <= 0 ||
      sync_tolerance_sec_ <= 0.0 || sensor_timeout_sec_ <= 0.0)
    {
      throw std::invalid_argument("queue depths and timeouts must be positive");
    }
    if (run_mode_ != "normal" && run_mode_ != "laser_fail" &&
      run_mode_ != "visual_fail")
    {
      throw std::invalid_argument(
              "run_mode must be normal, laser_fail, or visual_fail");
    }

    visual_queue_ = std::make_unique<lq::CausalQueue<lq::VisualQuality>>(
      static_cast<std::size_t>(queue_depth_));
    odom_queue_ = std::make_unique<lq::CausalQueue<PoseSample>>(
      static_cast<std::size_t>(queue_depth_));
    visual_localization_queue_ = std::make_unique<lq::CausalQueue<double>>(
      static_cast<std::size_t>(queue_depth_));
    laser_localization_queue_ = std::make_unique<lq::CausalQueue<double>>(
      static_cast<std::size_t>(queue_depth_));

    if (!output_csv_.empty()) {
      csv_.open(output_csv_);
      if (!csv_.is_open()) {
        throw std::runtime_error("cannot open output_csv: " + output_csv_);
      }
      csv_ <<
        "timestamp,x,y,coordinate_frame,q_laser_raw,q_visual_raw,q_laser,"
        "q_visual,q_fused,w_laser,w_visual,scene,p_corridor,p_hall,p_outdoor,"
        "illumination_quality,geometry_complexity,openness,strategy,"
        "calibration_domain_guard,fusion_state,laser_status,visual_status,image_delta_sec,odom_delta_sec,"
        "processing_ms,end_to_end_latency_ms\n";
    }

    metrics_publisher_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      metrics_topic_, 10);
    diagnostics_publisher_ =
      create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      diagnostics_topic_, 10);
    quality_map_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      quality_map_topic_, rclcpp::QoS(1).reliable().transient_local());
    if (publish_statistics_) {
      quality_stats_publisher_ = create_publisher<quality_navigation_msgs::msg::QualityGrid>(
        quality_stats_topic_, rclcpp::QoS(1).reliable().transient_local());
    }

    const rclcpp::QoS image_qos = configured_qos(
      static_cast<std::size_t>(image_qos_.depth),
      image_qos_.reliability, image_qos_.durability, "qos.image");
    const rclcpp::QoS laser_qos = configured_qos(
      static_cast<std::size_t>(laser_qos_.depth),
      laser_qos_.reliability, laser_qos_.durability, "qos.laser");
    const rclcpp::QoS odom_qos = configured_qos(
      static_cast<std::size_t>(odom_qos_.depth),
      odom_qos_.reliability, odom_qos_.durability, "qos.odom");
    const rclcpp::QoS localization_qos = configured_qos(
      static_cast<std::size_t>(localization_qos_.depth),
      localization_qos_.reliability, localization_qos_.durability,
      "qos.localization");
    if (image_transport_ == "raw") {
      image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, image_qos,
        std::bind(&RealtimeEvaluator::on_image, this, _1));
    } else {
      compressed_image_subscription_ =
        create_subscription<sensor_msgs::msg::CompressedImage>(
        compressed_image_topic_, image_qos,
        std::bind(&RealtimeEvaluator::on_compressed_image, this, _1));
    }
    if (laser_input_ == "scan") {
      scan_subscription_ = create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic_, laser_qos,
        std::bind(&RealtimeEvaluator::on_scan, this, _1));
    } else {
      cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        point_cloud_topic_, laser_qos,
        std::bind(&RealtimeEvaluator::on_point_cloud, this, _1));
    }
    odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, odom_qos,
      std::bind(&RealtimeEvaluator::on_odom, this, _1));
    visual_localization_subscription_ =
      create_subscription<nav_msgs::msg::Odometry>(
      visual_localization_topic_, localization_qos,
      std::bind(&RealtimeEvaluator::on_visual_localization, this, _1));
    laser_localization_subscription_ =
      create_subscription<nav_msgs::msg::Odometry>(
      laser_localization_topic_, localization_qos,
      std::bind(&RealtimeEvaluator::on_laser_localization, this, _1));

    RCLCPP_INFO(
      get_logger(),
      "started: image=%s laser=%s sync=%s model=%s map_frame=%s",
      image_transport_.c_str(), laser_input_.c_str(),
      lq::CausalQueue<int>::direction(),
      fusion_->model_loaded() ? "loaded" : "fallback", target_frame_.c_str());
  }

private:
  void on_visual_localization(
    const nav_msgs::msg::Odometry::SharedPtr message)
  {
    const double timestamp = message_time_seconds(*message, now().seconds());
    std::lock_guard<std::mutex> lock(cache_mutex_);
    visual_localization_queue_->push(timestamp, covariance_trace(*message));
  }

  void on_laser_localization(
    const nav_msgs::msg::Odometry::SharedPtr message)
  {
    const double timestamp = message_time_seconds(*message, now().seconds());
    std::lock_guard<std::mutex> lock(cache_mutex_);
    laser_localization_queue_->push(timestamp, covariance_trace(*message));
  }

  void process_image(const cv::Mat & source, double timestamp)
  {
    cv::Mat evaluated = source;
    if (run_mode_ == "visual_fail") {
      evaluated = lq::inject_visual_degradation(source, fault_severity_);
    }
    double covariance = std::numeric_limits<double>::infinity();
    double localization_delta = 0.0;
    bool localization_available = false;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      localization_available = visual_localization_queue_->match(
        timestamp, sensor_timeout_sec_, &covariance, &localization_delta);
    }
    std::string tracking_state = !localization_available ? "unknown" :
      (covariance <= tracking_covariance_limit_ ? "tracking" : "lost");
    if (run_mode_ == "visual_fail") {
      tracking_state = "lost";
      covariance = std::max(10.0, tracking_covariance_limit_ + 1.0);
    }
    lq::VisualQuality quality = visual_evaluator_->evaluate(
      evaluated, tracking_state, covariance);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    visual_queue_->push(timestamp, quality);
  }

  void on_image(const sensor_msgs::msg::Image::SharedPtr message)
  {
    try {
      const cv::Mat image = cv_bridge::toCvCopy(
        message, sensor_msgs::image_encodings::BGR8)->image;
      process_image(
        image, message_time_seconds(*message, now().seconds()));
    } catch (const cv_bridge::Exception & error) {
      lq::VisualQuality invalid;
      invalid.status = "cv_bridge_error";
      std::lock_guard<std::mutex> lock(cache_mutex_);
      visual_queue_->push(
        message_time_seconds(*message, now().seconds()), invalid);
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "raw image conversion failed: %s", error.what());
    }
  }

  void on_compressed_image(
    const sensor_msgs::msg::CompressedImage::SharedPtr message)
  {
    try {
      const cv::Mat encoded(
        1, static_cast<int>(message->data.size()), CV_8UC1,
        const_cast<unsigned char *>(message->data.data()));
      const cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
      if (image.empty()) {
        throw std::runtime_error("imdecode returned an empty image");
      }
      process_image(
        image, message_time_seconds(*message, now().seconds()));
    } catch (const std::exception & error) {
      lq::VisualQuality invalid;
      invalid.status = "compressed_decode_error";
      std::lock_guard<std::mutex> lock(cache_mutex_);
      visual_queue_->push(
        message_time_seconds(*message, now().seconds()), invalid);
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "compressed image conversion failed: %s", error.what());
    }
  }

  void on_odom(const nav_msgs::msg::Odometry::SharedPtr message)
  {
    PoseSample pose;
    pose.x = message->pose.pose.position.x;
    pose.y = message->pose.pose.position.y;
    pose.frame_id = message->header.frame_id.empty() ?
      "odom" : message->header.frame_id;
    if (pose.frame_id != target_frame_) {
      geometry_msgs::msg::PoseStamped input;
      input.header = message->header;
      input.pose = message->pose.pose;
      try {
        const geometry_msgs::msg::PoseStamped transformed =
          tf_buffer_->transform(
          input, target_frame_, tf2::durationFromSec(0.02));
        pose.x = transformed.pose.position.x;
        pose.y = transformed.pose.position.y;
        pose.frame_id = target_frame_;
      } catch (const tf2::TransformException & error) {
        ++tf_failures_;
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "pose transform %s -> %s unavailable: %s",
          input.header.frame_id.c_str(), target_frame_.c_str(), error.what());
      }
    }
    std::lock_guard<std::mutex> lock(cache_mutex_);
    odom_queue_->push(
      message_time_seconds(*message, now().seconds()), pose);
  }

  void on_scan(const sensor_msgs::msg::LaserScan::SharedPtr message)
  {
    std::vector<float> ranges = message->ranges;
    if (run_mode_ == "laser_fail") {
      lq::inject_laser_occlusion(
        ranges, fault_severity_, quality_config_.laser_close_range_m * 0.8);
    }
    lq::LaserQuality quality = lq::evaluate_laser(
      ranges, message->range_min, message->range_max,
      message->angle_min, message->angle_increment, quality_config_);
    process_laser(
      quality, message_time_seconds(*message, now().seconds()));
  }

  void on_point_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr message)
  {
    lq::LaserQuality quality;
    try {
      std::vector<cv::Point3f> points;
      points.reserve(
        static_cast<std::size_t>(message->width) * message->height);
      sensor_msgs::PointCloud2ConstIterator<float> x(*message, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y(*message, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z(*message, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        points.emplace_back(*x, *y, *z);
      }
      if (run_mode_ == "laser_fail") {
        lq::inject_point_cloud_occlusion(
          points, fault_severity_, quality_config_.laser_close_range_m * 0.8);
      }
      quality = lq::evaluate_point_cloud(points, quality_config_);
    } catch (const std::runtime_error & error) {
      quality.status = "pointcloud_field_error";
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "PointCloud2 conversion failed: %s", error.what());
    }
    process_laser(
      quality, message_time_seconds(*message, now().seconds()));
  }

  void process_laser(lq::LaserQuality laser, double timestamp)
  {
    const auto processing_start = std::chrono::steady_clock::now();
    lq::VisualQuality visual;
    PoseSample pose;
    double image_delta = std::numeric_limits<double>::quiet_NaN();
    double odom_delta = std::numeric_limits<double>::quiet_NaN();
    double laser_covariance = std::numeric_limits<double>::infinity();
    double localization_delta = 0.0;
    bool has_visual = false;
    bool has_pose = false;
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      has_visual = visual_queue_->match(
        timestamp, sync_tolerance_sec_, &visual, &image_delta);
      has_pose = odom_queue_->match(
        timestamp, sync_tolerance_sec_, &pose, &odom_delta);
      if (laser_localization_queue_->match(
          timestamp, sensor_timeout_sec_,
          &laser_covariance, &localization_delta) &&
        std::isfinite(laser_covariance))
      {
        const double covariance_score =
          std::exp(-std::max(0.0, laser_covariance) / 0.5);
        laser.quality *= 0.7 + 0.3 * covariance_score;
        if (laser_covariance > tracking_covariance_limit_) {
          laser.status = "localization_uncertain";
        }
      }
      visual_queue_->prune_before(timestamp - sensor_timeout_sec_);
      odom_queue_->prune_before(timestamp - sensor_timeout_sec_);
      laser_localization_queue_->prune_before(
        timestamp - sensor_timeout_sec_);
    }
    if (!has_visual) {
      visual = lq::VisualQuality{};
      visual.status = "sync_timeout";
      ++image_timeouts_;
    }
    if (!has_pose) {
      ++odom_timeouts_;
      if (odom_required_) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "odometry required but no causal sample is within tolerance");
      }
    }

    const lq::FusionResult fused = fusion_->update(laser, visual);
    const auto processing_end = std::chrono::steady_clock::now();
    const double processing_ms =
      std::chrono::duration<double, std::milli>(
      processing_end - processing_start).count();
    const double latency_ms = std::max(
      0.0, (now().seconds() - timestamp) * 1000.0);
    ++outputs_;
    publish(
      timestamp, pose, has_pose, laser, visual, fused,
      image_delta, odom_delta, processing_ms, latency_ms);
  }

  void publish(
    double timestamp,
    const PoseSample & pose,
    bool has_pose,
    const lq::LaserQuality & laser,
    const lq::VisualQuality & visual,
    const lq::FusionResult & fused,
    double image_delta,
    double odom_delta,
    double processing_ms,
    double latency_ms)
  {
    std_msgs::msg::Float32MultiArray metrics;
    // The first five fields are the v0.1 compatibility contract.
    metrics.data = {
      static_cast<float>(fused.q_laser),
      static_cast<float>(fused.q_visual),
      static_cast<float>(fused.q_fused),
      static_cast<float>(fused.w_laser),
      static_cast<float>(fused.w_visual),
      static_cast<float>(fused.context.scene_probabilities[0]),
      static_cast<float>(fused.context.scene_probabilities[1]),
      static_cast<float>(fused.context.scene_probabilities[2]),
      static_cast<float>(fused.context.illumination_quality),
      static_cast<float>(fused.context.geometry_complexity),
      static_cast<float>(fused.context.openness),
      laser.valid ? 1.0F : 0.0F,
      visual.valid ? 1.0F : 0.0F,
      static_cast<float>(image_delta),
      static_cast<float>(latency_ms)};
    metrics_publisher_->publish(metrics);

    if (has_pose && pose.frame_id == target_frame_) {
      grid_->update(pose.x, pose.y, fused.q_fused, timestamp);
      publish_quality_map(timestamp);
      if (quality_stats_publisher_) {
        statistics_->update(pose.x, pose.y, fused.q_fused, timestamp);
        if (lq::quality_statistics_due(
            timestamp, last_statistics_timestamp_, statistics_publish_period_sec_))
        {
          quality_stats_publisher_->publish(statistics_->snapshot(target_frame_, timestamp));
          last_statistics_timestamp_ = timestamp;
        }
      }
    }

    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "localization_quality/fusion";
    status.hardware_id = laser_input_ + "+" + image_transport_;
    status.level = fused.strategy == "both_degraded" ?
      diagnostic_msgs::msg::DiagnosticStatus::ERROR :
      (fused.strategy == "mlp_safety_override" ||
      fused.strategy == "rule_fallback" ?
      diagnostic_msgs::msg::DiagnosticStatus::WARN :
      diagnostic_msgs::msg::DiagnosticStatus::OK);
    status.message = fused.strategy + ":" + fused.state;
    status.values = {
      key_value("q_laser", number(fused.q_laser)),
      key_value("q_visual", number(fused.q_visual)),
      key_value("q_fused", number(fused.q_fused)),
      key_value("w_laser", number(fused.w_laser)),
      key_value("w_visual", number(fused.w_visual)),
      key_value("strategy", fused.strategy),
      key_value(
        "calibration_domain_guard",
        fused.calibration_domain_guard_applied ? "true" : "false"),
      key_value("fault_state", fused.state),
      key_value("model_loaded", fused.model_loaded ? "true" : "false"),
      key_value(
        "image_message_type",
        image_transport_ == "raw" ?
        "sensor_msgs/msg/Image" : "sensor_msgs/msg/CompressedImage"),
      key_value(
        "laser_message_type",
        laser_input_ == "scan" ?
        "sensor_msgs/msg/LaserScan" : "sensor_msgs/msg/PointCloud2"),
      key_value("qos_image", qos_summary(image_qos_)),
      key_value("qos_laser", qos_summary(laser_qos_)),
      key_value("qos_odom", qos_summary(odom_qos_)),
      key_value("qos_localization", qos_summary(localization_qos_)),
      key_value("odom_required", odom_required_ ? "true" : "false"),
      key_value("scene", lq::scene_type_name(fused.context.scene)),
      key_value("p_corridor", number(fused.context.scene_probabilities[0])),
      key_value("p_hall", number(fused.context.scene_probabilities[1])),
      key_value("p_outdoor", number(fused.context.scene_probabilities[2])),
      key_value("illumination_state", fused.context.illumination_state),
      key_value("illumination_level", number(fused.context.illumination_level)),
      key_value("illumination_quality", number(fused.context.illumination_quality)),
      key_value("geometry_complexity", number(fused.context.geometry_complexity)),
      key_value("openness", number(fused.context.openness)),
      key_value("directional_anisotropy", number(
          fused.context.directional_anisotropy)),
      key_value("laser_valid_rate", number(laser.valid_rate)),
      key_value("laser_coverage", number(laser.coverage_rate)),
      key_value("laser_occlusion", number(laser.close_ratio)),
      key_value("laser_direction_entropy", number(laser.direction_entropy)),
      key_value("laser_condition_number", number(laser.condition_number)),
      key_value("visual_blur_variance", number(visual.blur_variance)),
      key_value("visual_brightness", number(visual.brightness)),
      key_value("visual_contrast", number(visual.contrast)),
      key_value("visual_underexposed_ratio", number(visual.underexposed_ratio)),
      key_value("visual_overexposed_ratio", number(visual.overexposed_ratio)),
      key_value("visual_edge_density", number(visual.edge_density)),
      key_value("visual_features", std::to_string(visual.trackable_features)),
      key_value("visual_match_rate", number(visual.feature_match_rate)),
      key_value("visual_tracking_state", visual.tracking_state),
      key_value("sync_direction", lq::CausalQueue<int>::direction()),
      key_value("image_delta_sec", number(image_delta)),
      key_value("odom_delta_sec", number(odom_delta)),
      key_value("image_timeouts", std::to_string(image_timeouts_)),
      key_value("odom_timeouts", std::to_string(odom_timeouts_)),
      key_value("image_received", std::to_string(
          visual_queue_->statistics().received)),
      key_value("image_matched", std::to_string(
          visual_queue_->statistics().matched)),
      key_value("image_queue_drops", std::to_string(
          visual_queue_->statistics().queue_drops)),
      key_value("odom_received", std::to_string(
          odom_queue_->statistics().received)),
      key_value("odom_matched", std::to_string(
          odom_queue_->statistics().matched)),
      key_value("odom_queue_drops", std::to_string(
          odom_queue_->statistics().queue_drops)),
      key_value("outputs", std::to_string(outputs_)),
      key_value("processing_ms", number(processing_ms)),
      key_value("end_to_end_latency_ms", number(latency_ms)),
      key_value("coordinate_frame", has_pose ? pose.frame_id : "unavailable"),
      key_value("tf_failures", std::to_string(tf_failures_)),
      key_value("map_samples", std::to_string(grid_->accepted_samples()))};
    array.status.push_back(status);
    diagnostics_publisher_->publish(array);

    if (csv_.is_open()) {
      csv_ << std::setprecision(12) <<
        timestamp << ',' <<
        (has_pose ? number(pose.x) : "nan") << ',' <<
        (has_pose ? number(pose.y) : "nan") << ',' <<
        (has_pose ? pose.frame_id : "unavailable") << ',' <<
        laser.quality << ',' << visual.quality << ',' <<
        fused.q_laser << ',' << fused.q_visual << ',' << fused.q_fused << ',' <<
        fused.w_laser << ',' << fused.w_visual << ',' <<
        lq::scene_type_name(fused.context.scene) << ',' <<
        fused.context.scene_probabilities[0] << ',' <<
        fused.context.scene_probabilities[1] << ',' <<
        fused.context.scene_probabilities[2] << ',' <<
        fused.context.illumination_quality << ',' <<
        fused.context.geometry_complexity << ',' <<
        fused.context.openness << ',' << fused.strategy << ',' <<
        (fused.calibration_domain_guard_applied ? 1 : 0) << ',' <<
        fused.state << ',' << laser.status << ',' << visual.status << ',' <<
        number(image_delta) << ',' << number(odom_delta) << ',' <<
        processing_ms << ',' << latency_ms << '\n';
      csv_.flush();
    }
  }

  void publish_quality_map(double timestamp)
  {
    nav_msgs::msg::OccupancyGrid message;
    message.header.stamp = rclcpp::Time(
      static_cast<std::int64_t>(timestamp * 1e9));
    message.header.frame_id = target_frame_;
    message.info.map_load_time = message.header.stamp;
    message.info.resolution = static_cast<float>(grid_->config().resolution);
    message.info.width = grid_->config().width;
    message.info.height = grid_->config().height;
    message.info.origin.position.x = grid_->config().origin_x;
    message.info.origin.position.y = grid_->config().origin_y;
    message.info.origin.orientation.w = 1.0;
    message.data = grid_->occupancy_data();
    quality_map_publisher_->publish(message);
  }

  static std::string qos_summary(const QosSettings & settings)
  {
    return settings.reliability + "/" + settings.durability + "/" +
           std::to_string(settings.depth);
  }

  lq::QualityConfig quality_config_;
  std::unique_ptr<lq::VisualQualityEvaluator> visual_evaluator_;
  std::unique_ptr<lq::AdaptiveFusion> fusion_;
  std::unique_ptr<lq::QualityGrid> grid_;
  std::unique_ptr<lq::QualityStatisticsAccumulator> statistics_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<lq::CausalQueue<lq::VisualQuality>> visual_queue_;
  std::unique_ptr<lq::CausalQueue<PoseSample>> odom_queue_;
  std::unique_ptr<lq::CausalQueue<double>> visual_localization_queue_;
  std::unique_ptr<lq::CausalQueue<double>> laser_localization_queue_;
  std::mutex cache_mutex_;

  std::string scan_topic_;
  std::string point_cloud_topic_;
  std::string image_topic_;
  std::string compressed_image_topic_;
  std::string odom_topic_;
  std::string visual_localization_topic_;
  std::string laser_localization_topic_;
  std::string metrics_topic_;
  std::string diagnostics_topic_;
  std::string quality_map_topic_;
  std::string quality_stats_topic_;
  std::string image_transport_;
  std::string laser_input_;
  std::string target_frame_;
  std::string output_csv_;
  std::string run_mode_;
  QosSettings image_qos_;
  QosSettings laser_qos_;
  QosSettings odom_qos_;
  QosSettings localization_qos_;
  bool odom_required_ = true;
  bool publish_statistics_ = true;
  double statistics_publish_period_sec_ = 0.5;
  double last_statistics_timestamp_ = -std::numeric_limits<double>::infinity();
  double sync_tolerance_sec_ = 0.1;
  double sensor_timeout_sec_ = 0.5;
  double tracking_covariance_limit_ = 0.5;
  double fault_severity_ = 1.0;
  int queue_depth_ = 50;
  std::size_t image_timeouts_ = 0;
  std::size_t odom_timeouts_ = 0;
  std::size_t outputs_ = 0;
  std::size_t tf_failures_ = 0;
  std::ofstream csv_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr metrics_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostics_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr quality_map_publisher_;
  rclcpp::Publisher<quality_navigation_msgs::msg::QualityGrid>::SharedPtr
    quality_stats_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
    compressed_image_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
    visual_localization_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
    laser_localization_subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<RealtimeEvaluator>());
  } catch (const std::exception & error) {
    std::cerr << "realtime_evaluator: " << error.what() << '\n';
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
