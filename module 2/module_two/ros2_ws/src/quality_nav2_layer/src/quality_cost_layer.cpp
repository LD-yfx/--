#include "quality_nav2_layer/quality_cost_layer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "pluginlib/class_list_macros.hpp"
#include "tf2/exceptions.h"
#include "tf2/time.h"

namespace quality_nav2_layer
{
void QualityCostLayer::onInitialize()
{
  const auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("quality cost layer cannot lock its lifecycle node");
  }
  clock_ = node->get_clock();
  logger_ = node->get_logger();
  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("topic", rclcpp::ParameterValue("/localization_quality/navigation_cost"));
  declareParameter("stale_timeout", rclcpp::ParameterValue(5.0));
  declareParameter("stale_cost", rclcpp::ParameterValue(200));
  declareParameter("unknown_cost", rclcpp::ParameterValue(160));
  declareParameter("combination_method", rclcpp::ParameterValue("saturating_add"));
  std::string topic, combination;
  int stale_cost, unknown_cost;
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".topic", topic);
  node->get_parameter(name_ + ".stale_timeout", stale_timeout_);
  node->get_parameter(name_ + ".stale_cost", stale_cost);
  node->get_parameter(name_ + ".unknown_cost", unknown_cost);
  node->get_parameter(name_ + ".combination_method", combination);
  if (topic.empty() || !std::isfinite(stale_timeout_) || stale_timeout_ <= 0.0 ||
    stale_cost < 0 || stale_cost > kMaxSoftCost ||
    unknown_cost < 0 || unknown_cost > kMaxSoftCost)
  {
    throw std::invalid_argument(
            "quality layer: topic must be nonempty, stale_timeout finite > 0, costs in [0,252]");
  }
  if (combination == "saturating_add") {
    combination_ = Combination::SaturatingAdd;
  } else if (combination == "max") {
    combination_ = Combination::Maximum;
  } else {
    throw std::invalid_argument("quality combination_method must be saturating_add or max");
  }
  stale_cost_ = static_cast<uint8_t>(stale_cost);
  unknown_cost_ = static_cast<uint8_t>(unknown_cost);
  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_;
  subscription_ = node->create_subscription<nav2_msgs::msg::Costmap>(
    topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
    [this](nav2_msgs::msg::Costmap::ConstSharedPtr message) {
      receiveMap(std::move(message));
    }, options);
  current_ = true;
  RCLCPP_INFO(logger_, "Quality layer subscribes to %s (%s, unknown=%u, stale=%u)",
    topic.c_str(), combination.c_str(), unsigned(unknown_cost_), unsigned(stale_cost_));
}

void QualityCostLayer::receiveMap(nav2_msgs::msg::Costmap::ConstSharedPtr message)
{
  std::string reason;
  const bool valid = validateMessage(*message, reason);
  const auto now_ns = clock_->now().nanoseconds();
  const auto stamp_ns = stampNanoseconds(message->header.stamp);
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (!valid || stamp_ns > now_ns + kFutureToleranceNs) {
    invalid_source_ = true;
    if (valid) {
      reason = "timestamp is more than 0.5 seconds in the future";
    }
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
      "Rejecting quality map: %s; conservative fallback is active", reason.c_str());
    return;
  }
  if (latest_map_ && stamp_ns <= stampNanoseconds(latest_map_->header.stamp)) {
    // A delayed/latched replay must not refresh the wall-clock reception timer.
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
      "Ignoring duplicate/out-of-order quality map timestamp; freshness was not extended");
    return;
  }
  latest_map_ = std::move(message);
  received_at_ = std::chrono::steady_clock::now();
  invalid_source_ = false;
}

void QualityCostLayer::updateBounds(
  double, double, double, double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }
  const auto * master = layered_costmap_->getCostmap();
  // Nav2 resets these bounds before replaying all layers. Rebuild the whole master
  // so map shrinkage, rolling origins, and reduced penalties do not leave ghosts.
  *min_x = std::min(*min_x, master->getOriginX());
  *min_y = std::min(*min_y, master->getOriginY());
  *max_x = std::max(*max_x, master->getOriginX() +
    master->getSizeInCellsX() * master->getResolution());
  *max_y = std::max(*max_y, master->getOriginY() +
    master->getSizeInCellsY() * master->getResolution());
}

bool QualityCostLayer::resolveTransform(
  const std::string & source_frame, PlanarTransform & transform)
{
  const auto & global_frame = layered_costmap_->getGlobalFrameID();
  if (source_frame == global_frame) {
    transform = PlanarTransform{};
    return true;
  }
  try {
    // Transform master cell centers into the quality map's header frame. This is
    // intentionally the latest spatial alignment of two persistent map frames.
    const auto stamped = tf_->lookupTransform(source_frame, global_frame, tf2::TimePointZero);
    const auto & t = stamped.transform.translation;
    const auto & q = stamped.transform.rotation;
    const int64_t tf_ns = stampNanoseconds(stamped.header.stamp);
    if (tf_ns != 0 && isStale(clock_->now().nanoseconds(), tf_ns, 0.0, stale_timeout_)) {
      throw std::runtime_error("latest map-frame transform is stale");
    }
    if (!std::isfinite(t.x) || !std::isfinite(t.y) || !std::isfinite(q.z) ||
      !std::isfinite(q.w) || !std::isfinite(q.x) || !std::isfinite(q.y) ||
      std::abs(q.x) > 1e-4 || std::abs(q.y) > 1e-4 ||
      std::abs(q.z * q.z + q.w * q.w - 1.0) > 1e-3)
    {
      throw std::runtime_error("map-frame transform is nonplanar or nonfinite");
    }
    const double yaw = std::atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z);
    transform = PlanarTransform{t.x, t.y, std::cos(yaw), std::sin(yaw)};
    return true;
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
      "Quality layer TF unavailable (%s); last projection plus stale floor is used",
      error.what());
    return false;
  }
}

void QualityCostLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }
  nav2_msgs::msg::Costmap::ConstSharedPtr map;
  std::chrono::steady_clock::time_point received;
  bool invalid;
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    map = latest_map_;
    received = received_at_;
    invalid = invalid_source_;
  }
  bool stale = invalid || !map;
  if (map) {
    const double reception_age = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - received).count();
    stale = stale || isStale(clock_->now().nanoseconds(),
      stampNanoseconds(map->header.stamp), reception_age, stale_timeout_);
  }
  PlanarTransform transform;
  if (map && resolveTransform(map->header.frame_id, transform)) {
    last_projected_map_ = map;
    last_transform_ = transform;
  } else {
    stale = true;
    map = last_projected_map_;
    transform = last_transform_;
  }
  if (stale) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
      "Quality source missing, stale, invalid, or untransformable: applying cost floor %u",
      unsigned(stale_cost_));
  }
  min_i = std::max(0, min_i);
  min_j = std::max(0, min_j);
  max_i = std::min(max_i, static_cast<int>(master.getSizeInCellsX()));
  max_j = std::min(max_j, static_cast<int>(master.getSizeInCellsY()));
  for (int j = min_j; j < max_j; ++j) {
    for (int i = min_i; i < max_i; ++i) {
      const auto master_cost = master.getCost(i, j);
      if (master_cost > kMaxSoftCost) {
        continue;
      }
      uint8_t quality_cost = unknown_cost_;
      if (map) {
        double world_x, world_y;
        master.mapToWorld(i, j, world_x, world_y);
        const double source_x = transform.cosine * world_x - transform.sine * world_y +
          transform.x;
        const double source_y = transform.sine * world_x + transform.cosine * world_y +
          transform.y;
        quality_cost = sampleCost(*map, source_x, source_y, unknown_cost_);
      }
      if (stale) {
        quality_cost = std::max(quality_cost, stale_cost_);
      }
      master.setCost(i, j, combineCost(master_cost, quality_cost, combination_));
    }
  }
  // A complete conservative map has been produced. Marking current_ false would
  // prevent Nav2 from planning at all, defeating this explicitly chosen fallback.
  current_ = true;
}

void QualityCostLayer::reset()
{
  // Recovery must not erase environmental quality evidence. Full bounds rebuild
  // the costs on the next update, while source/arrival timestamps remain intact.
  current_ = true;
}
}  // namespace quality_nav2_layer

PLUGINLIB_EXPORT_CLASS(quality_nav2_layer::QualityCostLayer, nav2_costmap_2d::Layer)
