#include "localization_quality/quality_core.hpp"
#include "localization_quality/ros_parameters.hpp"

#include "cv_bridge/cv_bridge.h"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lq = localization_quality;

namespace
{

bool valid_mode(const std::string & mode)
{
  return mode == "normal" || mode == "laser_fail" || mode == "visual_fail";
}

double record_time_seconds(const rosbag2_storage::SerializedBagMessage & message)
{
  return static_cast<double>(message.time_stamp) / 1e9;
}

template<typename MessageT>
double message_time_seconds(const MessageT & message, double fallback)
{
  const double timestamp = static_cast<double>(message.header.stamp.sec) +
    static_cast<double>(message.header.stamp.nanosec) / 1e9;
  return timestamp > 0.0 ? timestamp : fallback;
}

template<typename MessageT>
std::shared_ptr<MessageT> deserialize(
  const std::shared_ptr<rosbag2_storage::SerializedBagMessage> & bag_message)
{
  auto message = std::make_shared<MessageT>();
  rclcpp::Serialization<MessageT> serializer;
  rclcpp::SerializedMessage serialized(*bag_message->serialized_data);
  serializer.deserialize_message(&serialized, message.get());
  return message;
}

void write_csv_header(std::ofstream & stream)
{
  stream <<
    "timestamp,x,y,q_laser_raw,q_visual_raw,q_laser,q_visual,q_fused,"
    "w_laser,w_visual,laser_valid_rate,laser_close_ratio,laser_coverage_rate,"
    "laser_occlusion_penalty,visual_blur_variance,visual_brightness,"
    "visual_contrast,visual_blur_score,visual_brightness_score,"
    "visual_contrast_score,laser_valid,visual_valid,fusion_state,laser_status,"
    "visual_status,mode,visual_age_sec,odom_age_sec,processing_ms,"
    "sync_direction,strategy,calibration_domain_guard,scene,p_corridor,p_hall,p_outdoor,"
    "illumination_level,illumination_quality,illumination_state,"
    "geometry_complexity,openness,directional_anisotropy,laser_direction_entropy,"
    "laser_corridor_degeneracy,laser_abrupt_change_rate,laser_condition_number,"
    "laser_linearity,laser_planarity,laser_scattering,visual_underexposed_ratio,"
    "visual_overexposed_ratio,visual_edge_density,visual_trackable_features,"
    "visual_feature_match_rate,visual_motion_blur,visual_tracking_state,"
    "visual_localization_covariance,laser_localization_covariance,"
    "coordinate_frame\n";
}

struct LaserObservation
{
  double timestamp = 0.0;
  double processing_ms = 0.0;
  lq::LaserQuality quality;
};

struct VisualObservation
{
  double timestamp = 0.0;
  double processing_ms = 0.0;
  lq::VisualQuality quality;
};

struct OdomObservation
{
  double timestamp = 0.0;
  double x = 0.0;
  double y = 0.0;
  std::string frame_id = "odom";
};

struct LocalizationObservation
{
  double timestamp = 0.0;
  double covariance_trace = std::numeric_limits<double>::infinity();
};

template<typename ObservationT>
const ObservationT * causal_observation(
  const std::vector<ObservationT> & observations, double timestamp)
{
  if (observations.empty()) {
    return nullptr;
  }
  const auto next = std::upper_bound(
    observations.begin(), observations.end(), timestamp,
    [](double value, const ObservationT & observation) {
      return value < observation.timestamp;
    });
  if (next == observations.begin()) {
    return nullptr;
  }
  return &(*(next - 1));
}

}  // namespace

class OfflineEvaluator : public rclcpp::Node
{
public:
  OfflineEvaluator(
    const std::string & bag_path,
    const std::string & run_mode,
    const std::string & output_path)
  : Node("offline_evaluator"), run_mode_(run_mode), output_path_(output_path)
  {
    if (!valid_mode(run_mode_)) {
      throw std::invalid_argument(
              "run mode must be one of: normal, laser_fail, visual_fail");
    }

    quality_config_ = lq::declare_quality_parameters(*this);
    visual_evaluator_ = std::make_unique<lq::VisualQualityEvaluator>(
      quality_config_);
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
    image_transport_ = declare_parameter<std::string>("input.image", "raw");
    laser_input_ = declare_parameter<std::string>("input.laser", "scan");
    odom_required_ = declare_parameter<bool>("input.odom_required", true);
    sync_tolerance_sec_ = declare_parameter<double>("sync.tolerance_sec", 0.10);
    tracking_covariance_limit_ = declare_parameter<double>(
      "quality.visual.tracking_covariance_limit", 0.50);
    fault_severity_ = declare_parameter<double>("fault_severity", 1.0);
    if (sync_tolerance_sec_ <= 0.0) {
      throw std::invalid_argument("sync_tolerance_sec must be positive");
    }
    if (image_transport_ != "raw" && image_transport_ != "compressed") {
      throw std::invalid_argument("input.image must be raw or compressed");
    }
    if (laser_input_ != "scan" && laser_input_ != "pointcloud2") {
      throw std::invalid_argument("input.laser must be scan or pointcloud2");
    }
    if (fault_severity_ < 0.0 || fault_severity_ > 1.0) {
      throw std::invalid_argument("fault_severity must be in [0, 1]");
    }

    RCLCPP_INFO(get_logger(), "bag: %s", bag_path.c_str());
    RCLCPP_INFO(get_logger(), "mode: %s", run_mode_.c_str());
    RCLCPP_INFO(get_logger(), "output: %s", output_path_.c_str());
    process_bag(bag_path);
  }

private:
  void validate_topics(rosbag2_cpp::Reader & reader)
  {
    bool found_laser = false;
    bool found_image = false;
    bool found_odom = false;
    for (const auto & topic : reader.get_all_topics_and_types()) {
      if (laser_input_ == "scan" && topic.name == scan_topic_) {
        found_laser = topic.type == "sensor_msgs/msg/LaserScan";
      } else if (laser_input_ == "pointcloud2" &&
        topic.name == point_cloud_topic_)
      {
        found_laser = topic.type == "sensor_msgs/msg/PointCloud2";
      } else if (image_transport_ == "raw" && topic.name == image_topic_) {
        found_image = topic.type == "sensor_msgs/msg/Image";
      } else if (image_transport_ == "compressed" &&
        topic.name == compressed_image_topic_)
      {
        found_image = topic.type == "sensor_msgs/msg/CompressedImage";
      } else if (topic.name == odom_topic_) {
        found_odom = topic.type == "nav_msgs/msg/Odometry";
      }
    }
    if (!found_laser || !found_image || (odom_required_ && !found_odom)) {
      throw std::runtime_error(
              "bag does not contain configured input topics with expected types");
    }
  }

  VisualObservation process_image(
    const std::shared_ptr<sensor_msgs::msg::Image> & message,
    double record_timestamp)
  {
    VisualObservation observation;
    observation.timestamp = message_time_seconds(*message, record_timestamp);
    const auto start = std::chrono::steady_clock::now();
    try {
      const cv::Mat source = cv_bridge::toCvShare(message, "bgr8")->image;
      const cv::Mat evaluated_image = run_mode_ == "visual_fail" ?
        lq::inject_visual_degradation(source, fault_severity_) : source;
      observation.quality = visual_evaluator_->evaluate(evaluated_image);
    } catch (const cv_bridge::Exception & error) {
      observation.quality.status = "cv_bridge_error";
      ++image_conversion_errors_;
      RCLCPP_WARN(get_logger(), "image conversion failed: %s", error.what());
    }
    const auto end = std::chrono::steady_clock::now();
    observation.processing_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
    return observation;
  }

  VisualObservation process_compressed_image(
    const std::shared_ptr<sensor_msgs::msg::CompressedImage> & message,
    double record_timestamp)
  {
    VisualObservation observation;
    observation.timestamp = message_time_seconds(*message, record_timestamp);
    const auto start = std::chrono::steady_clock::now();
    try {
      const cv::Mat encoded(
        1, static_cast<int>(message->data.size()), CV_8UC1,
        const_cast<unsigned char *>(message->data.data()));
      const cv::Mat source = cv::imdecode(encoded, cv::IMREAD_COLOR);
      if (source.empty()) {
        throw std::runtime_error("imdecode returned an empty image");
      }
      const cv::Mat evaluated_image = run_mode_ == "visual_fail" ?
        lq::inject_visual_degradation(source, fault_severity_) : source;
      observation.quality = visual_evaluator_->evaluate(evaluated_image);
    } catch (const std::exception & error) {
      observation.quality.status = "compressed_decode_error";
      ++image_conversion_errors_;
      RCLCPP_WARN(get_logger(), "image conversion failed: %s", error.what());
    }
    const auto end = std::chrono::steady_clock::now();
    observation.processing_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
    return observation;
  }

  OdomObservation process_odom(
    const std::shared_ptr<nav_msgs::msg::Odometry> & message,
    double record_timestamp)
  {
    OdomObservation observation;
    observation.timestamp = message_time_seconds(*message, record_timestamp);
    observation.x = message->pose.pose.position.x;
    observation.y = message->pose.pose.position.y;
    observation.frame_id = message->header.frame_id.empty() ?
      "odom" : message->header.frame_id;
    return observation;
  }

  LocalizationObservation process_localization(
    const std::shared_ptr<nav_msgs::msg::Odometry> & message,
    double record_timestamp)
  {
    LocalizationObservation observation;
    observation.timestamp = message_time_seconds(*message, record_timestamp);
    observation.covariance_trace = 0.0;
    for (std::size_t index : {0U, 7U, 14U, 21U, 28U, 35U}) {
      const double value = message->pose.covariance[index];
      if (!std::isfinite(value) || value < 0.0) {
        observation.covariance_trace =
          std::numeric_limits<double>::infinity();
        break;
      }
      observation.covariance_trace += value;
    }
    return observation;
  }

  LaserObservation process_scan(
    const std::shared_ptr<sensor_msgs::msg::LaserScan> & message,
    double record_timestamp)
  {
    LaserObservation observation;
    observation.timestamp = message_time_seconds(*message, record_timestamp);
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> ranges = message->ranges;
    if (run_mode_ == "laser_fail") {
      lq::inject_laser_occlusion(
        ranges, fault_severity_, quality_config_.laser_close_range_m * 0.8);
    }
    observation.quality = lq::evaluate_laser(
      ranges, message->range_min, message->range_max,
      message->angle_min, message->angle_increment, quality_config_);
    const auto end = std::chrono::steady_clock::now();
    observation.processing_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
    return observation;
  }

  LaserObservation process_point_cloud(
    const std::shared_ptr<sensor_msgs::msg::PointCloud2> & message,
    double record_timestamp)
  {
    LaserObservation observation;
    observation.timestamp = message_time_seconds(*message, record_timestamp);
    const auto start = std::chrono::steady_clock::now();
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
      observation.quality = lq::evaluate_point_cloud(points, quality_config_);
    } catch (const std::runtime_error & error) {
      observation.quality.status = "pointcloud_field_error";
      RCLCPP_WARN(get_logger(), "point cloud conversion failed: %s", error.what());
    }
    const auto end = std::chrono::steady_clock::now();
    observation.processing_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
    return observation;
  }

  void write_results(
    const std::vector<LaserObservation> & scans,
    const std::vector<VisualObservation> & images,
    const std::vector<OdomObservation> & odometry,
    const std::vector<LocalizationObservation> & visual_localization,
    const std::vector<LocalizationObservation> & laser_localization,
    std::ofstream & csv)
  {
    write_csv_header(csv);
    fusion_->reset();

    for (const auto & scan : scans) {
      const auto * visual_candidate = causal_observation(images, scan.timestamp);
      const auto * odom_candidate = causal_observation(odometry, scan.timestamp);
      const double visual_age = visual_candidate == nullptr ?
        std::numeric_limits<double>::quiet_NaN() :
        scan.timestamp - visual_candidate->timestamp;
      const double odom_age = odom_candidate == nullptr ?
        std::numeric_limits<double>::quiet_NaN() :
        scan.timestamp - odom_candidate->timestamp;
      VisualObservation missing_visual;
      missing_visual.quality.status = visual_candidate == nullptr ?
        "missing_image" : "sync_timeout";
      const VisualObservation * visual =
        visual_candidate != nullptr && visual_age <= sync_tolerance_sec_ ?
        visual_candidate : &missing_visual;
      const bool odom_valid =
        odom_candidate != nullptr && odom_age <= sync_tolerance_sec_;
      if (visual == &missing_visual || !odom_valid) {
        if (visual_candidate == nullptr || odom_candidate == nullptr) {
          ++skipped_missing_cache_;
        } else {
          ++skipped_out_of_sync_;
        }
      }

      lq::VisualQuality visual_quality = visual->quality;
      const auto * visual_pose = causal_observation(
        visual_localization, scan.timestamp);
      if (visual_pose != nullptr &&
        scan.timestamp - visual_pose->timestamp <= sync_tolerance_sec_)
      {
        visual_quality = lq::apply_visual_localization_status(
          visual_quality,
          visual_pose->covariance_trace <= tracking_covariance_limit_ ?
          "tracking" : "lost",
          visual_pose->covariance_trace);
      }
      // The explicit visual_fail regression mode represents the complete
      // visual-localization channel and injects matching lost-state covariance.
      if (run_mode_ == "visual_fail") {
        visual_quality = lq::apply_visual_localization_status(
          visual_quality, "lost",
          std::max(10.0, tracking_covariance_limit_ + 1.0));
      }
      lq::LaserQuality laser_quality = scan.quality;
      const auto * laser_pose = causal_observation(
        laser_localization, scan.timestamp);
      if (laser_pose != nullptr &&
        scan.timestamp - laser_pose->timestamp <= sync_tolerance_sec_ &&
        std::isfinite(laser_pose->covariance_trace))
      {
        laser_quality.quality *= 0.7 + 0.3 * std::exp(
          -std::max(0.0, laser_pose->covariance_trace) / 0.5);
        if (laser_pose->covariance_trace > tracking_covariance_limit_) {
          laser_quality.status = "localization_uncertain";
        }
      }
      const lq::FusionResult fused = fusion_->update(
        laser_quality, visual_quality);
      const double processing_ms = scan.processing_ms + visual->processing_ms;
      const double x = odom_valid ? odom_candidate->x :
        std::numeric_limits<double>::quiet_NaN();
      const double y = odom_valid ? odom_candidate->y :
        std::numeric_limits<double>::quiet_NaN();

      csv << std::fixed << std::setprecision(6)
        << scan.timestamp << ',' << x << ',' << y << ','
        << laser_quality.quality << ',' << visual_quality.quality << ','
        << fused.q_laser << ',' << fused.q_visual << ',' << fused.q_fused << ','
        << fused.w_laser << ',' << fused.w_visual << ','
        << scan.quality.valid_rate << ',' << scan.quality.close_ratio << ','
        << scan.quality.coverage_rate << ',' << scan.quality.occlusion_penalty << ','
        << visual_quality.blur_variance << ',' << visual_quality.brightness << ','
        << visual_quality.contrast << ',' << visual_quality.blur_score << ','
        << visual_quality.brightness_score << ',' << visual_quality.contrast_score << ','
        << (laser_quality.valid ? 1 : 0) << ',' << (visual_quality.valid ? 1 : 0) << ','
        << fused.state << ',' << laser_quality.status << ',' << visual_quality.status << ','
        << run_mode_ << ',' << visual_age << ',' << odom_age << ','
        << processing_ms << ",causal_past," << fused.strategy << ','
        << (fused.calibration_domain_guard_applied ? 1 : 0) << ','
        << lq::scene_type_name(fused.context.scene) << ','
        << fused.context.scene_probabilities[0] << ','
        << fused.context.scene_probabilities[1] << ','
        << fused.context.scene_probabilities[2] << ','
        << fused.context.illumination_level << ','
        << fused.context.illumination_quality << ','
        << fused.context.illumination_state << ','
        << fused.context.geometry_complexity << ','
        << fused.context.openness << ','
        << fused.context.directional_anisotropy << ','
        << scan.quality.direction_entropy << ','
        << scan.quality.corridor_degeneracy << ','
        << scan.quality.abrupt_change_rate << ','
        << scan.quality.condition_number << ','
        << scan.quality.linearity << ','
        << scan.quality.planarity << ','
        << scan.quality.scattering << ','
        << visual_quality.underexposed_ratio << ','
        << visual_quality.overexposed_ratio << ','
        << visual_quality.edge_density << ','
        << visual_quality.trackable_features << ','
        << visual_quality.feature_match_rate << ','
        << visual_quality.motion_blur << ','
        << visual_quality.tracking_state << ','
        << visual_quality.localization_covariance << ','
        << (laser_pose == nullptr ?
        std::numeric_limits<double>::quiet_NaN() :
        laser_pose->covariance_trace) << ','
        << (odom_valid ? odom_candidate->frame_id : "unavailable") << '\n';

      ++output_count_;
      state_counts_[fused.state]++;
      processing_ms_sum_ += processing_ms;
      processing_ms_max_ = std::max(processing_ms_max_, processing_ms);
    }
  }

  void process_bag(const std::string & bag_path)
  {
    rosbag2_cpp::Reader reader;
    reader.open(bag_path);
    validate_topics(reader);
    std::ofstream csv(output_path_);
    if (!csv.is_open()) {
      throw std::runtime_error("cannot open output CSV: " + output_path_);
    }

    std::vector<LaserObservation> scans;
    std::vector<VisualObservation> images;
    std::vector<OdomObservation> odometry;
    std::vector<LocalizationObservation> visual_localization;
    std::vector<LocalizationObservation> laser_localization;

    while (reader.has_next()) {
      auto bag_message = reader.read_next();
      const double record_timestamp = record_time_seconds(*bag_message);
      if (image_transport_ == "raw" &&
        bag_message->topic_name == image_topic_)
      {
        images.push_back(process_image(
            deserialize<sensor_msgs::msg::Image>(bag_message), record_timestamp));
      } else if (image_transport_ == "compressed" &&
        bag_message->topic_name == compressed_image_topic_)
      {
        images.push_back(process_compressed_image(
            deserialize<sensor_msgs::msg::CompressedImage>(
              bag_message), record_timestamp));
      } else if (bag_message->topic_name == odom_topic_) {
        odometry.push_back(process_odom(
            deserialize<nav_msgs::msg::Odometry>(bag_message), record_timestamp));
      } else if (bag_message->topic_name == visual_localization_topic_) {
        visual_localization.push_back(process_localization(
            deserialize<nav_msgs::msg::Odometry>(
              bag_message), record_timestamp));
      } else if (bag_message->topic_name == laser_localization_topic_) {
        laser_localization.push_back(process_localization(
            deserialize<nav_msgs::msg::Odometry>(
              bag_message), record_timestamp));
      } else if (laser_input_ == "scan" &&
        bag_message->topic_name == scan_topic_)
      {
        scans.push_back(process_scan(
            deserialize<sensor_msgs::msg::LaserScan>(bag_message), record_timestamp));
      } else if (laser_input_ == "pointcloud2" &&
        bag_message->topic_name == point_cloud_topic_)
      {
        scans.push_back(process_point_cloud(
            deserialize<sensor_msgs::msg::PointCloud2>(
              bag_message), record_timestamp));
      }
    }

    const auto by_timestamp = [](const auto & left, const auto & right) {
        return left.timestamp < right.timestamp;
      };
    std::sort(scans.begin(), scans.end(), by_timestamp);
    std::sort(images.begin(), images.end(), by_timestamp);
    std::sort(odometry.begin(), odometry.end(), by_timestamp);
    std::sort(visual_localization.begin(), visual_localization.end(), by_timestamp);
    std::sort(laser_localization.begin(), laser_localization.end(), by_timestamp);
    scan_count_ = scans.size();
    image_count_ = images.size();
    odom_count_ = odometry.size();
    write_results(
      scans, images, odometry, visual_localization, laser_localization, csv);
    csv.close();

    if (output_count_ == 0) {
      throw std::runtime_error(
              "no synchronized output rows were produced; check topics and sync_tolerance_sec");
    }
    const double matched_rate = static_cast<double>(output_count_) /
      static_cast<double>(scan_count_);
    RCLCPP_INFO(
      get_logger(),
      "messages scan=%zu image=%zu odom=%zu; output=%zu (%.1f%% of scans)",
      scan_count_, image_count_, odom_count_, output_count_, matched_rate * 100.0);
    RCLCPP_INFO(
      get_logger(),
      "skipped missing_stream=%zu out_of_sync=%zu image_errors=%zu; processing mean=%.3f ms max=%.3f ms",
      skipped_missing_cache_, skipped_out_of_sync_, image_conversion_errors_,
      processing_ms_sum_ / static_cast<double>(output_count_), processing_ms_max_);
    for (const auto & state : state_counts_) {
      RCLCPP_INFO(
        get_logger(), "fusion state %s: %zu", state.first.c_str(), state.second);
    }
  }

  lq::QualityConfig quality_config_;
  std::unique_ptr<lq::VisualQualityEvaluator> visual_evaluator_;
  std::unique_ptr<lq::AdaptiveFusion> fusion_;
  std::string run_mode_;
  std::string output_path_;
  std::string scan_topic_;
  std::string point_cloud_topic_;
  std::string image_topic_;
  std::string compressed_image_topic_;
  std::string odom_topic_;
  std::string visual_localization_topic_;
  std::string laser_localization_topic_;
  std::string image_transport_;
  std::string laser_input_;
  bool odom_required_ = true;
  double sync_tolerance_sec_ = 0.25;
  double fault_severity_ = 1.0;
  double tracking_covariance_limit_ = 0.50;

  std::size_t scan_count_ = 0;
  std::size_t image_count_ = 0;
  std::size_t odom_count_ = 0;
  std::size_t output_count_ = 0;
  std::size_t skipped_missing_cache_ = 0;
  std::size_t skipped_out_of_sync_ = 0;
  std::size_t image_conversion_errors_ = 0;
  double processing_ms_sum_ = 0.0;
  double processing_ms_max_ = 0.0;
  std::map<std::string, std::size_t> state_counts_;
};

int main(int argc, char * argv[])
{
  // ORB may otherwise use a machine-dependent parallel schedule.  A fixed
  // thread count makes archive extraction byte-reproducible across reruns.
  cv::setNumThreads(1);
  cv::setRNGSeed(20260727);
  rclcpp::init(argc, argv);
  const auto arguments = rclcpp::remove_ros_arguments(argc, argv);
  if (arguments.size() < 2 || arguments.size() > 4) {
    RCLCPP_ERROR(
      rclcpp::get_logger("offline_evaluator"),
      "usage: offline_evaluator BAG_PATH [normal|laser_fail|visual_fail] [OUTPUT_CSV]");
    rclcpp::shutdown();
    return 2;
  }

  const std::string bag_path = arguments[1];
  const std::string mode = arguments.size() >= 3 ? arguments[2] : "normal";
  const std::string output = arguments.size() >= 4 ?
    arguments[3] : "quality_results_" + mode + ".csv";

  try {
    auto evaluator = std::make_shared<OfflineEvaluator>(bag_path, mode, output);
    (void)evaluator;
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      rclcpp::get_logger("offline_evaluator"), "evaluation failed: %s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
