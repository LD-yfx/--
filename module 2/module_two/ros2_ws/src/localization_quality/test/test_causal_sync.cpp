#include "localization_quality/causal_sync.hpp"

#include <gtest/gtest.h>

namespace lq = localization_quality;

TEST(CausalSync, NeverUsesFutureSamplesAndBoundsQueue)
{
  lq::CausalQueue<int> queue(3);
  queue.push(1.0, 10);
  queue.push(3.0, 30);
  queue.push(2.0, 20);
  int value = 0;
  double delta = 0.0;
  ASSERT_TRUE(queue.match(2.5, 1.0, &value, &delta));
  EXPECT_EQ(value, 20);
  EXPECT_DOUBLE_EQ(delta, 0.5);
  queue.push(4.0, 40);
  EXPECT_EQ(queue.size(), 3U);
  EXPECT_EQ(queue.statistics().queue_drops, 1U);
  EXPECT_FALSE(queue.match(1.5, 1.0, &value, &delta));
  EXPECT_STREQ(lq::CausalQueue<int>::direction(), "causal_past");
}
