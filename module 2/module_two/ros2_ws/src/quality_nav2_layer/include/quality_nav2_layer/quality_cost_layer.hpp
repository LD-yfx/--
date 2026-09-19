#ifndef QUALITY_NAV2_LAYER__QUALITY_COST_LAYER_HPP_
#define QUALITY_NAV2_LAYER__QUALITY_COST_LAYER_HPP_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>

#include "nav2_costmap_2d/layer.hpp"
#include "nav2_msgs/msg/costmap.hpp"
#include "quality_nav2_layer/cost_semantics.hpp"

namespace quality_nav2_layer
{
class QualityCostLayer : public nav2_costmap_2d::Layer
{
public:
  void onInitialize() override;
  void updateBounds(double, double, double, double *, double *, double *, double *) override;
  void updateCosts(nav2_costmap_2d::Costmap2D &, int, int, int, int) override;
  void reset() override;
  bool isClearable() override {return false;}

private:
  struct PlanarTransform
  {
    double x{0.0}, y{0.0}, cosine{1.0}, sine{0.0};
  };
  void receiveMap(nav2_msgs::msg::Costmap::ConstSharedPtr message);
  bool resolveTransform(const std::string & source_frame, PlanarTransform & transform);

  rclcpp::Subscription<nav2_msgs::msg::Costmap>::SharedPtr subscription_;
  std::mutex map_mutex_;
  nav2_msgs::msg::Costmap::ConstSharedPtr latest_map_;
  std::chrono::steady_clock::time_point received_at_{};
  bool invalid_source_{false};
  // Used only by the serialized costmap update loop, never by the subscription.
  nav2_msgs::msg::Costmap::ConstSharedPtr last_projected_map_;
  PlanarTransform last_transform_;
  double stale_timeout_{5.0};
  uint8_t stale_cost_{200};
  uint8_t unknown_cost_{160};
  Combination combination_{Combination::SaturatingAdd};
};
}  // namespace quality_nav2_layer
#endif  // QUALITY_NAV2_LAYER__QUALITY_COST_LAYER_HPP_
