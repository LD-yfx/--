#include "localization_quality/quality_statistics.hpp"
#include "gtest/gtest.h"

#include <limits>

namespace lq = localization_quality;

static lq::QualityGrid small_grid()
{
  lq::QualityGridConfig config;
  config.width = 3;
  config.height = 2;
  config.resolution = 1.0;
  config.origin_x = -1.0;
  config.origin_y = -1.0;
  return lq::QualityGrid(config);
}

TEST(QualityStatistics, SnapshotKeepsPerCellObservationsAndUnbiasedVariance)
{
  auto grid = small_grid();
  ASSERT_TRUE(grid.update(-0.5, -0.5, 0.0, 1.25));
  ASSERT_TRUE(grid.update(-0.5, -0.5, 1.0, 2.5));
  ASSERT_TRUE(grid.update(1.5, 0.5, 0.8, 3.75));
  const auto result = lq::make_quality_statistics(grid, "map", 10.5);
  EXPECT_EQ(result.statistics_mode, "cumulative");
  EXPECT_EQ(result.header.frame_id, "map");
  EXPECT_EQ(result.header.stamp.sec, 10);
  EXPECT_EQ(result.header.stamp.nanosec, 500000000U);
  EXPECT_EQ(result.info.width, 3U);
  EXPECT_EQ(result.info.height, 2U);
  EXPECT_DOUBLE_EQ(result.info.origin.position.x, -1.0);
  EXPECT_DOUBLE_EQ(result.info.origin.orientation.w, 1.0);
  ASSERT_EQ(result.indices.size(), 2U);
  EXPECT_EQ(result.indices[0], 0U);
  EXPECT_EQ(result.indices[1], 5U);
  EXPECT_FLOAT_EQ(result.mean[0], 0.5F);
  EXPECT_FLOAT_EQ(result.variance[0], 0.5F);
  EXPECT_EQ(result.sample_count[0], 2U);
  EXPECT_EQ(result.last_observed[0].sec, 2);
  EXPECT_EQ(result.last_observed[0].nanosec, 500000000U);
  EXPECT_EQ(result.last_observed[1].sec, 3);
  EXPECT_EQ(result.last_observed[1].nanosec, 750000000U);
  EXPECT_FLOAT_EQ(result.variance[1], 0.0F);
  EXPECT_EQ(result.mean.size(), result.indices.size());
  EXPECT_EQ(result.variance.size(), result.indices.size());
  EXPECT_EQ(result.sample_count.size(), result.indices.size());
  EXPECT_EQ(result.last_observed.size(), result.indices.size());
}

TEST(QualityStatistics, EmptySnapshotReplacesPreviouslyKnownCellsAfterReset)
{
  auto grid = small_grid();
  grid.update(-0.5, -0.5, 0.5, 10.0);
  ASSERT_EQ(lq::make_quality_statistics(grid, "odom", 10.0).indices.size(), 1U);
  grid.reset();
  const auto result = lq::make_quality_statistics(grid, "odom", 0.0);
  EXPECT_TRUE(result.indices.empty());
  EXPECT_TRUE(result.mean.empty());
  EXPECT_TRUE(result.variance.empty());
  EXPECT_TRUE(result.sample_count.empty());
  EXPECT_TRUE(result.last_observed.empty());
}

TEST(QualityStatistics, RejectsInvalidOrFutureObservationTime)
{
  auto grid = small_grid();
  grid.update(-0.5, -0.5, 0.5, 10.0);
  EXPECT_THROW(lq::make_quality_statistics(grid, "odom", 9.0), std::invalid_argument);
  grid.update(-0.5, -0.5, 0.5, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(lq::make_quality_statistics(grid, "odom", 11.0), std::invalid_argument);
  EXPECT_THROW(lq::quality_statistics_time(-1.0), std::invalid_argument);
  EXPECT_THROW(lq::quality_statistics_time(std::numeric_limits<double>::infinity()),
    std::invalid_argument);
}

TEST(QualityStatistics, SensorClockControlsPublicationPeriod)
{
  EXPECT_TRUE(lq::quality_statistics_due(0.0,
    -std::numeric_limits<double>::infinity(), 0.5));
  EXPECT_FALSE(lq::quality_statistics_due(10.25, 10.0, 0.5));
  EXPECT_TRUE(lq::quality_statistics_due(10.5, 10.0, 0.5));
  EXPECT_TRUE(lq::quality_statistics_due(0.0, 10.0, 0.5));
  EXPECT_THROW(lq::quality_statistics_due(10.0, 9.0, 0.0), std::invalid_argument);
}

TEST(QualityStatisticsWindow, ExpiresCellsAndRecoversAfterQualityImproves)
{
  const auto config = small_grid().config();
  lq::QualityStatisticsAccumulator statistics(config, 10.0, 256);
  statistics.update(-0.5, -0.5, 0.0, 1.0);
  statistics.update(1.5, 0.5, 0.2, 2.0);
  statistics.update(-0.5, -0.5, 1.0, 11.0);
  auto result = statistics.snapshot("map", 11.0);
  ASSERT_EQ(result.indices.size(), 2U);
  EXPECT_EQ(result.statistics_mode, "windowed");
  EXPECT_FLOAT_EQ(result.mean[0], 0.5F);  // boundary observation at t=1 is retained
  result = statistics.snapshot("map", 12.5);
  ASSERT_EQ(result.indices.size(), 1U);
  EXPECT_EQ(result.indices[0], 0U);
  EXPECT_FLOAT_EQ(result.mean[0], 1.0F);
  EXPECT_FLOAT_EQ(result.variance[0], 0.0F);
  EXPECT_EQ(result.sample_count[0], 1U);
  EXPECT_EQ(result.last_observed[0].sec, 11);
  EXPECT_TRUE(statistics.snapshot("map", 22.0).indices.empty());
}

TEST(QualityStatisticsWindow, CapacityRetainsMostRecentSamples)
{
  lq::QualityStatisticsAccumulator statistics(small_grid().config(), 10.0, 2);
  statistics.update(-0.5, -0.5, 0.0, 1.0);
  statistics.update(-0.5, -0.5, 0.4, 2.0);
  statistics.update(-0.5, -0.5, 0.8, 3.0);
  const auto result = statistics.snapshot("map", 3.0);
  ASSERT_EQ(result.indices.size(), 1U);
  EXPECT_EQ(result.sample_count[0], 2U);
  EXPECT_FLOAT_EQ(result.mean[0], 0.6F);
  EXPECT_FLOAT_EQ(result.variance[0], 0.08F);
}

TEST(QualityStatisticsWindow, CumulativeModeMatchesOriginalGrid)
{
  auto original = small_grid();
  lq::QualityStatisticsAccumulator statistics(original.config(), 0.0, 1);
  for (int i = 0; i < 5; ++i) {
    original.update(-0.5, -0.5, 0.2 * i, i + 1.0);
    statistics.update(-0.5, -0.5, 0.2 * i, i + 1.0);
  }
  const auto result = statistics.snapshot("map", 100.0);
  const auto cell = original.cell(0, 0);
  ASSERT_EQ(result.indices.size(), 1U);
  EXPECT_EQ(result.statistics_mode, "cumulative");
  EXPECT_EQ(result.sample_count[0], cell.count);
  EXPECT_FLOAT_EQ(result.mean[0], static_cast<float>(cell.mean));
  EXPECT_FLOAT_EQ(result.variance[0], static_cast<float>(cell.variance));
  EXPECT_EQ(result.last_observed[0].sec, 5);
}

TEST(QualityStatisticsWindow, ClockRewindClearsOnlyStatisticsAccumulator)
{
  auto original = small_grid();
  original.update(1.5, 0.5, 0.2, 100.0);
  lq::QualityStatisticsAccumulator statistics(original.config(), 10.0, 256);
  statistics.update(1.5, 0.5, 0.2, 100.0);
  statistics.update(-0.5, -0.5, 0.8, 1.0);
  const auto result = statistics.snapshot("map", 1.0);
  ASSERT_EQ(result.indices.size(), 1U);
  EXPECT_EQ(result.indices[0], 0U);
  EXPECT_EQ(result.last_observed[0].sec, 1);
  EXPECT_EQ(original.cell(2, 1).count, 1U);
}

TEST(QualityStatisticsWindow, RejectsInvalidGeometryWindowAndSamples)
{
  const auto config = small_grid().config();
  EXPECT_THROW(lq::QualityStatisticsAccumulator(config, -1.0, 256), std::invalid_argument);
  EXPECT_THROW(lq::QualityStatisticsAccumulator(config, 10.0, 0), std::invalid_argument);
  lq::QualityStatisticsAccumulator statistics(config, 10.0, 256);
  EXPECT_FALSE(statistics.update(-0.5, -0.5, 1.1, 1.0));
  EXPECT_FALSE(statistics.update(-0.5, -0.5, 0.8, -1.0));
  EXPECT_FALSE(statistics.update(100.0, 100.0, 0.8, 1.0));
  EXPECT_FALSE(statistics.update(-0.5, -0.5, 0.8,
    std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(statistics.snapshot("map", 1.0).indices.empty());
}
