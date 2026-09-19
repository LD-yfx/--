#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>

namespace localization_quality
{

struct SyncStatistics
{
  std::size_t received = 0;
  std::size_t matched = 0;
  std::size_t missing = 0;
  std::size_t outside_tolerance = 0;
  std::size_t queue_drops = 0;
};

template<typename ValueT>
class CausalQueue
{
public:
  struct Item
  {
    double timestamp = 0.0;
    ValueT value;
  };

  explicit CausalQueue(std::size_t capacity)
  : capacity_(capacity)
  {
    if (capacity_ == 0) {
      throw std::invalid_argument("causal queue capacity must be positive");
    }
  }

  void push(double timestamp, const ValueT & value)
  {
    ++statistics_.received;
    Item item{timestamp, value};
    const auto position = std::upper_bound(
      items_.begin(), items_.end(), timestamp,
      [](double target, const Item & candidate) {
        return target < candidate.timestamp;
      });
    items_.insert(position, item);
    while (items_.size() > capacity_) {
      items_.pop_front();
      ++statistics_.queue_drops;
    }
  }

  bool match(double pivot, double tolerance, ValueT * value, double * delta)
  {
    if (value == nullptr || delta == nullptr) {
      throw std::invalid_argument("causal queue output pointers must not be null");
    }
    const Item * candidate = nullptr;
    for (auto iterator = items_.rbegin(); iterator != items_.rend(); ++iterator) {
      if (iterator->timestamp <= pivot) {
        candidate = &*iterator;
        break;
      }
    }
    if (candidate == nullptr) {
      ++statistics_.missing;
      return false;
    }
    *delta = pivot - candidate->timestamp;
    if (*delta > tolerance) {
      ++statistics_.outside_tolerance;
      return false;
    }
    *value = candidate->value;
    ++statistics_.matched;
    return true;
  }

  void prune_before(double timestamp)
  {
    while (!items_.empty() && items_.front().timestamp < timestamp) {
      items_.pop_front();
    }
  }

  std::size_t size() const
  {
    return items_.size();
  }

  const SyncStatistics & statistics() const
  {
    return statistics_;
  }

  static const char * direction()
  {
    return "causal_past";
  }

private:
  std::size_t capacity_;
  std::deque<Item> items_;
  SyncStatistics statistics_;
};

}  // namespace localization_quality
