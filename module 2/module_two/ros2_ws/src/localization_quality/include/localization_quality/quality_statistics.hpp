#pragma once

#include "localization_quality/quality_core.hpp"
#include "quality_navigation_msgs/msg/quality_grid.hpp"

#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace localization_quality
{

// Convert the source sensor clock without consulting wall time. A double
// timestamp retains the precision available from the original grid API.
inline builtin_interfaces::msg::Time quality_statistics_time(double seconds)
{
  if (!std::isfinite(seconds) || seconds < 0.0 ||
    seconds >= static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0)
  {
    throw std::invalid_argument("quality statistics timestamp is outside ROS Time range");
  }
  std::int64_t whole = static_cast<std::int64_t>(std::floor(seconds));
  std::int64_t fraction = static_cast<std::int64_t>(
    std::llround((seconds - static_cast<double>(whole)) * 1e9));
  if (fraction == 1000000000LL) {
    ++whole;
    fraction = 0;
  }
  if (whole > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument("quality statistics timestamp rounds outside ROS Time range");
  }
  builtin_interfaces::msg::Time result;
  result.sec = static_cast<std::int32_t>(whole);
  result.nanosec = static_cast<std::uint32_t>(fraction);
  return result;
}

inline bool quality_statistics_due(double sensor_time, double last_published, double period)
{
  if (!std::isfinite(sensor_time) || sensor_time < 0.0 ||
    !std::isfinite(period) || period <= 0.0)
  {
    throw std::invalid_argument("invalid quality statistics publish time or period");
  }
  return !std::isfinite(last_published) || sensor_time < last_published ||
         sensor_time - last_published >= period;
}

inline quality_navigation_msgs::msg::QualityGrid make_quality_statistics(
  const QualityGrid & grid, const std::string & frame_id, double sensor_time)
{
  const auto & config = grid.config();
  const auto cells = static_cast<std::uint64_t>(config.width) * config.height;
  if (frame_id.empty() || cells > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("quality statistics require a frame and uint32 cell indices");
  }
  quality_navigation_msgs::msg::QualityGrid result;
  result.header.stamp = quality_statistics_time(sensor_time);
  result.header.frame_id = frame_id;
  result.info.map_load_time = result.header.stamp;
  result.info.resolution = static_cast<float>(config.resolution);
  result.info.width = config.width;
  result.info.height = config.height;
  result.info.origin.position.x = config.origin_x;
  result.info.origin.position.y = config.origin_y;
  result.info.origin.orientation.w = 1.0;
  result.statistics_mode = "cumulative";
  for (std::uint32_t y = 0; y < config.height; ++y) {
    for (std::uint32_t x = 0; x < config.width; ++x) {
      const auto cell = grid.cell(x, y);
      if (cell.count == 0) {
        continue;
      }
      if (!std::isfinite(cell.last_update) || cell.last_update > sensor_time) {
        throw std::invalid_argument("quality cell time is invalid or newer than the snapshot");
      }
      result.indices.push_back(y * config.width + x);
      result.mean.push_back(static_cast<float>(cell.mean));
      result.variance.push_back(static_cast<float>(cell.variance));
      result.sample_count.push_back(cell.count);
      result.last_observed.push_back(quality_statistics_time(cell.last_update));
    }
  }
  return result;
}

// Statistics for the new interface are independent of the legacy cumulative
// QualityGrid. This permits recovery after an environment change without
// changing either the original quality map or the fusion model.
class QualityStatisticsAccumulator
{
public:
  QualityStatisticsAccumulator(
    const QualityGridConfig & config, double window_sec, std::size_t max_samples_per_cell)
  : config_(config), window_sec_(window_sec), max_samples_(max_samples_per_cell)
  {
    if (!std::isfinite(window_sec_) || window_sec_ < 0.0 || max_samples_ == 0 ||
      !std::isfinite(config_.resolution) || config_.resolution <= 0.0 ||
      !std::isfinite(config_.origin_x) || !std::isfinite(config_.origin_y) ||
      config_.width == 0 || config_.height == 0 ||
      static_cast<std::uint64_t>(config_.width) * config_.height >
      std::numeric_limits<std::uint32_t>::max())
    {
      throw std::invalid_argument("invalid quality statistics geometry or window limits");
    }
  }

  bool update(double x, double y, double quality, double timestamp)
  {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(quality) ||
      quality < 0.0 || quality > 1.0 || !std::isfinite(timestamp) || timestamp < 0.0 ||
      timestamp >= static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0)
    {
      return false;
    }
    reset_on_rewind(timestamp);
    const double grid_x = std::floor((x - config_.origin_x) / config_.resolution);
    const double grid_y = std::floor((y - config_.origin_y) / config_.resolution);
    if (grid_x < 0.0 || grid_y < 0.0 ||
      grid_x >= config_.width || grid_y >= config_.height)
    {
      return false;
    }
    const auto index = static_cast<std::uint32_t>(grid_y) * config_.width +
      static_cast<std::uint32_t>(grid_x);
    auto & cell = cells_[index];
    if (window_sec_ > 0.0) {
      discard_expired(cell, timestamp);
      while (cell.samples.size() >= max_samples_) {
        cell.samples.pop_front();
      }
      cell.samples.push_back({timestamp, quality});
    } else {
      ++cell.count;
      const double difference = quality - cell.mean;
      cell.mean += difference / static_cast<double>(cell.count);
      cell.m2 += difference * (quality - cell.mean);
    }
    cell.last_observed = timestamp;
    return true;
  }

  quality_navigation_msgs::msg::QualityGrid snapshot(
    const std::string & frame_id, double timestamp)
  {
    if (frame_id.empty()) {
      throw std::invalid_argument("quality statistics require a nonempty frame");
    }
    const auto stamp = quality_statistics_time(timestamp);
    reset_on_rewind(timestamp);
    quality_navigation_msgs::msg::QualityGrid result;
    result.header.stamp = stamp;
    result.header.frame_id = frame_id;
    result.info.map_load_time = stamp;
    result.info.resolution = static_cast<float>(config_.resolution);
    result.info.width = config_.width;
    result.info.height = config_.height;
    result.info.origin.position.x = config_.origin_x;
    result.info.origin.position.y = config_.origin_y;
    result.info.origin.orientation.w = 1.0;
    result.statistics_mode = window_sec_ > 0.0 ? "windowed" : "cumulative";
    for (auto iterator = cells_.begin(); iterator != cells_.end(); ) {
      auto & cell = iterator->second;
      if (window_sec_ > 0.0) {
        discard_expired(cell, timestamp);
        if (cell.samples.empty()) {
          iterator = cells_.erase(iterator);
          continue;
        }
        cell.count = 0;
        cell.mean = 0.0;
        cell.m2 = 0.0;
        for (const auto & sample : cell.samples) {
          ++cell.count;
          const double difference = sample.quality - cell.mean;
          cell.mean += difference / static_cast<double>(cell.count);
          cell.m2 += difference * (sample.quality - cell.mean);
        }
      }
      result.indices.push_back(iterator->first);
      result.mean.push_back(static_cast<float>(cell.mean));
      result.variance.push_back(static_cast<float>(
        cell.count > 1 ? cell.m2 / static_cast<double>(cell.count - 1) : 0.0));
      result.sample_count.push_back(cell.count);
      result.last_observed.push_back(quality_statistics_time(cell.last_observed));
      ++iterator;
    }
    return result;
  }

private:
  struct Sample
  {
    double timestamp;
    double quality;
  };

  struct Cell
  {
    std::deque<Sample> samples;
    std::uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    double last_observed = 0.0;
  };

  void discard_expired(Cell & cell, double timestamp)
  {
    while (!cell.samples.empty() &&
      cell.samples.front().timestamp < timestamp - window_sec_)
    {
      cell.samples.pop_front();
    }
  }

  void reset_on_rewind(double timestamp)
  {
    if (timestamp < last_sensor_time_) {
      cells_.clear();
    }
    last_sensor_time_ = timestamp;
  }

  QualityGridConfig config_;
  double window_sec_;
  std::size_t max_samples_;
  double last_sensor_time_ = -std::numeric_limits<double>::infinity();
  std::map<std::uint32_t, Cell> cells_;
};

}  // namespace localization_quality
