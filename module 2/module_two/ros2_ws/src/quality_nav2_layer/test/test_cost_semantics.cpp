#include <cmath>
#include <limits>
#include <string>

#include "gtest/gtest.h"
#include "quality_nav2_layer/cost_semantics.hpp"

namespace q = quality_nav2_layer;

nav2_msgs::msg::Costmap makeMap()
{
  nav2_msgs::msg::Costmap map;
  map.header.frame_id = "odom";
  map.header.stamp.sec = 10;
  map.metadata.resolution = 0.5;
  map.metadata.size_x = 3;
  map.metadata.size_y = 2;
  map.metadata.origin.position.x = -2.0;
  map.metadata.origin.position.y = 1.0;
  map.metadata.origin.orientation.w = 1.0;
  map.data = {0, 25, 252, 255, 100, 180};
  return map;
}

TEST(Combination, PreservesEveryReservedMasterValue)
{
  for (int master = 253; master <= 255; ++master) {
    for (int penalty = 0; penalty <= 252; ++penalty) {
      EXPECT_EQ(q::combineCost(master, penalty, q::Combination::SaturatingAdd), master);
      EXPECT_EQ(q::combineCost(master, penalty, q::Combination::Maximum), master);
    }
  }
}

TEST(Combination, BoundedMonotoneAndIdentityForAllSoftCosts)
{
  for (int master = 0; master <= 252; ++master) {
    int previous = master;
    EXPECT_EQ(q::combineCost(master, 0, q::Combination::SaturatingAdd), master);
    for (int penalty = 0; penalty <= 252; ++penalty) {
      const auto combined = q::combineCost(master, penalty, q::Combination::SaturatingAdd);
      EXPECT_LE(combined, 252);
      EXPECT_GE(combined, master);
      EXPECT_GE(combined, previous);
      previous = combined;
    }
  }
  EXPECT_EQ(q::combineCost(170, 160, q::Combination::Maximum), 170);
  EXPECT_EQ(q::combineCost(170, 160, q::Combination::SaturatingAdd), 252);
}

TEST(Coordinates, SamplesNegativeOriginAndHalfOpenBounds)
{
  const auto map = makeMap();
  EXPECT_EQ(q::sampleCost(map, -1.75, 1.25, 160), 0);
  EXPECT_EQ(q::sampleCost(map, -0.75, 1.25, 160), 252);
  EXPECT_EQ(q::sampleCost(map, -1.25, 1.75, 160), 100);
  EXPECT_EQ(q::sampleCost(map, -2.01, 1.25, 160), 160);
  EXPECT_EQ(q::sampleCost(map, -0.5, 1.25, 160), 160);
  EXPECT_EQ(q::sampleCost(map, -1.0, 2.0, 160), 160);
  EXPECT_EQ(q::sampleCost(map, -1.75, 1.75, 160), 160);
  EXPECT_EQ(q::sampleCost(map, std::numeric_limits<double>::infinity(), 1, 160), 160);
}

TEST(Coordinates, SupportsRotatedOriginsAndReplacementGeometry)
{
  auto map = makeMap();
  map.metadata.origin.orientation.z = std::sqrt(0.5);
  map.metadata.origin.orientation.w = std::sqrt(0.5);
  // Local (0.75, 0.25) -> source (-2.25, 1.75), second column of first row.
  EXPECT_EQ(q::sampleCost(map, -2.25, 1.75, 160), 25);
  map.metadata.size_x = 1;
  map.metadata.size_y = 1;
  map.metadata.resolution = 1.0;
  map.metadata.origin.position.x = 10.0;
  map.metadata.origin.position.y = 20.0;
  map.data = {80};
  EXPECT_EQ(q::sampleCost(map, 9.5, 20.5, 160), 80);
  EXPECT_EQ(q::sampleCost(map, -2.25, 1.75, 160), 160);
}

TEST(Validation, AcceptsOnlyCompletePlanarQualityGrids)
{
  std::string reason;
  auto map = makeMap();
  EXPECT_TRUE(q::validateMessage(map, reason));
  map.data.pop_back();
  EXPECT_FALSE(q::validateMessage(map, reason));
  map = makeMap();
  map.data[1] = 254;
  EXPECT_FALSE(q::validateMessage(map, reason));
  map.data[1] = 253;
  EXPECT_FALSE(q::validateMessage(map, reason));
  map = makeMap();
  map.metadata.resolution = 0;
  EXPECT_FALSE(q::validateMessage(map, reason));
  map.metadata.resolution = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(q::validateMessage(map, reason));
  map = makeMap();
  map.metadata.origin.orientation.w = 0;
  EXPECT_FALSE(q::validateMessage(map, reason));
  map = makeMap();
  map.metadata.origin.orientation.x = 0.1;
  EXPECT_FALSE(q::validateMessage(map, reason));
  map = makeMap();
  map.metadata.origin.position.x = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(q::validateMessage(map, reason));
  map = makeMap();
  map.header.frame_id.clear();
  EXPECT_FALSE(q::validateMessage(map, reason));
  map = makeMap();
  map.header.stamp.nanosec = 1000000000;
  EXPECT_FALSE(q::validateMessage(map, reason));
  map = makeMap();
  map.metadata.size_x = std::numeric_limits<uint32_t>::max();
  map.metadata.size_y = std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(q::validateMessage(map, reason));
}

TEST(Freshness, RequiresSourceTimeAndReceptionTimeToBeFresh)
{
  EXPECT_FALSE(q::isStale(12000000000LL, 10000000000LL, 1.0, 5.0));
  EXPECT_TRUE(q::isStale(20000000000LL, 10000000000LL, 0.0, 5.0));
  EXPECT_TRUE(q::isStale(12000000000LL, 10000000000LL, 6.0, 5.0));
  EXPECT_TRUE(q::isStale(1000000000LL, 10000000000LL, 0.0, 5.0));
  EXPECT_TRUE(q::isStale(12000000000LL, 10000000000LL, -1.0, 5.0));
}
