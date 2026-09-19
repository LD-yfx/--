#ifndef QUALITY_NAV2_LAYER__COST_SEMANTICS_HPP_
#define QUALITY_NAV2_LAYER__COST_SEMANTICS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "nav2_msgs/msg/costmap.hpp"

namespace quality_nav2_layer
{
constexpr uint8_t kMaxSoftCost = 252;
constexpr uint8_t kUnknown = 255;
constexpr int64_t kFutureToleranceNs = 500000000;

enum class Combination { SaturatingAdd, Maximum };

inline uint8_t combineCost(uint8_t master, uint8_t quality, Combination method)
{
  // 253: inscribed inflation, 254: lethal obstacle, 255: unknown space.
  if (master > kMaxSoftCost) {
    return master;
  }
  quality = std::min(quality, kMaxSoftCost);
  if (method == Combination::Maximum) {
    return std::max(master, quality);
  }
  return static_cast<uint8_t>(std::min<unsigned>(kMaxSoftCost, master + quality));
}

inline int64_t stampNanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

inline bool validateMessage(const nav2_msgs::msg::Costmap & map, std::string & reason)
{
  const auto & m = map.metadata;
  const auto & p = m.origin.position;
  const auto & q = m.origin.orientation;
  if (map.header.frame_id.empty()) {
    reason = "empty frame_id";
  } else if (map.header.stamp.sec < 0 || map.header.stamp.nanosec >= 1000000000U) {
    reason = "invalid message timestamp";
  } else if (!std::isfinite(m.resolution) || m.resolution <= 0.0f) {
    reason = "resolution must be finite and positive";
  } else if (m.size_x == 0 || m.size_y == 0 ||
    static_cast<uint64_t>(m.size_x) * m.size_y != map.data.size())
  {
    reason = "size/data length mismatch or empty map";
  } else if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
    !std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) || !std::isfinite(q.w))
  {
    reason = "nonfinite map origin";
  } else if (std::abs(q.x) > 1e-6 || std::abs(q.y) > 1e-6 ||
    std::abs(q.z * q.z + q.w * q.w - 1.0) > 1e-3)
  {
    reason = "origin must contain a normalized planar quaternion";
  } else if (std::any_of(map.data.begin(), map.data.end(), [](uint8_t c) {
      return c == 253 || c == 254;
    }))
  {
    reason = "quality map cannot encode inscribed or lethal obstacles (253/254)";
  } else {
    reason.clear();
    return true;
  }
  return false;
}

// Inverse of origin translation/rotation: source-frame position -> row-major cell.
inline bool worldToIndex(
  const nav2_msgs::msg::Costmap & map, double x, double y, std::size_t & index)
{
  if (!std::isfinite(x) || !std::isfinite(y)) {
    return false;
  }
  const auto & m = map.metadata;
  const auto & q = m.origin.orientation;
  const double yaw = std::atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z);
  const double dx = x - m.origin.position.x;
  const double dy = y - m.origin.position.y;
  const double grid_x = (std::cos(yaw) * dx + std::sin(yaw) * dy) / m.resolution;
  const double grid_y = (-std::sin(yaw) * dx + std::cos(yaw) * dy) / m.resolution;
  if (!std::isfinite(grid_x) || !std::isfinite(grid_y) ||
    grid_x < 0.0 || grid_y < 0.0 || grid_x >= m.size_x || grid_y >= m.size_y)
  {
    return false;
  }
  index = static_cast<std::size_t>(std::floor(grid_y)) * m.size_x +
    static_cast<std::size_t>(std::floor(grid_x));
  return index < map.data.size();
}

inline uint8_t sampleCost(
  const nav2_msgs::msg::Costmap & map, double x, double y, uint8_t unknown_cost)
{
  std::size_t index = 0;
  if (!worldToIndex(map, x, y, index) || map.data[index] == kUnknown) {
    return unknown_cost;
  }
  return map.data[index];
}

inline bool isStale(
  int64_t now_ns, int64_t source_ns, double reception_age_s, double stale_timeout_s)
{
  // Reject clock reversals and future timestamps instead of making them look fresh.
  if (now_ns < source_ns - kFutureToleranceNs || reception_age_s < 0.0) {
    return true;
  }
  const double source_age_s = static_cast<double>(now_ns - source_ns) * 1e-9;
  return source_age_s > stale_timeout_s || reception_age_s > stale_timeout_s;
}
}  // namespace quality_nav2_layer
#endif  // QUALITY_NAV2_LAYER__COST_SEMANTICS_HPP_
