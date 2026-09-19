#include "localization_quality/quality_core.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace lq = localization_quality;

TEST(LaserQuality, ScoresHealthyAndInvalidScans)
{
  const lq::QualityConfig config;
  const std::vector<float> healthy(360, 2.0F);
  const auto healthy_result = lq::evaluate_laser(healthy, 0.1, 30.0, config);
  EXPECT_TRUE(healthy_result.valid);
  EXPECT_NEAR(healthy_result.quality, 1.0, 1e-9);
  EXPECT_NEAR(healthy_result.coverage_rate, 1.0, 1e-9);

  const std::vector<float> invalid(
    360, std::numeric_limits<float>::infinity());
  const auto invalid_result = lq::evaluate_laser(invalid, 0.1, 30.0, config);
  EXPECT_FALSE(invalid_result.valid);
  EXPECT_DOUBLE_EQ(invalid_result.quality, 0.0);
  EXPECT_EQ(invalid_result.status, "no_valid_returns");
}

TEST(LaserQuality, DetectsRawOcclusionInjection)
{
  const lq::QualityConfig config;
  std::vector<float> ranges(360, 2.0F);
  lq::inject_laser_occlusion(ranges, 1.0, 0.4);
  const auto result = lq::evaluate_laser(ranges, 0.1, 30.0, config);
  EXPECT_TRUE(result.valid);
  EXPECT_NEAR(result.close_ratio, 1.0, 1e-9);
  EXPECT_NEAR(result.quality, config.laser_occlusion_floor, 1e-9);
  EXPECT_EQ(result.status, "occluded");
}

TEST(VisualQuality, SeparatesTexturedAndDarkImages)
{
  const lq::QualityConfig config;
  cv::Mat checkerboard(240, 320, CV_8UC3);
  for (int row = 0; row < checkerboard.rows; ++row) {
    for (int column = 0; column < checkerboard.cols; ++column) {
      const unsigned char value = ((row / 8 + column / 8) % 2) ? 255 : 0;
      checkerboard.at<cv::Vec3b>(row, column) = cv::Vec3b(value, value, value);
    }
  }
  const auto healthy = lq::evaluate_visual(checkerboard, config);
  EXPECT_TRUE(healthy.valid);
  EXPECT_GT(healthy.quality, 0.8);

  const cv::Mat dark = cv::Mat::zeros(240, 320, CV_8UC3);
  const auto failed = lq::evaluate_visual(dark, config);
  EXPECT_TRUE(failed.valid);
  EXPECT_LE(failed.quality, config.failure_enter_threshold);

  const cv::Mat injected = lq::inject_visual_degradation(checkerboard, 1.0);
  const auto injected_result = lq::evaluate_visual(injected, config);
  EXPECT_LE(injected_result.quality, config.failure_enter_threshold);
}

TEST(AdaptiveFusion, HandlesNormalSingleFailureAndDoubleFailure)
{
  const lq::QualityConfig config;
  lq::AdaptiveFusion fusion(config);
  lq::LaserQuality laser;
  laser.valid = true;
  laser.quality = 0.9;
  lq::VisualQuality visual;
  visual.valid = true;
  visual.quality = 0.8;

  const auto normal = fusion.update(laser, visual);
  EXPECT_EQ(normal.state, "normal");
  EXPECT_NEAR(normal.w_laser + normal.w_visual, 1.0, 1e-12);
  EXPECT_GT(normal.w_laser, normal.w_visual);
  EXPECT_NEAR(
    normal.q_fused,
    normal.w_laser * normal.q_laser + normal.w_visual * normal.q_visual,
    1e-12);

  lq::AdaptiveFusion laser_failure_fusion(config);
  laser.quality = 0.1;
  const auto laser_failure = laser_failure_fusion.update(laser, visual);
  EXPECT_EQ(laser_failure.state, "laser_degraded");
  EXPECT_NEAR(laser_failure.w_visual, 0.9, 1e-12);

  laser.valid = false;
  visual.valid = false;
  const auto both_invalid = fusion.update(laser, visual);
  EXPECT_EQ(both_invalid.state, "invalid_both");
  EXPECT_DOUBLE_EQ(both_invalid.q_fused, 0.0);
  EXPECT_NEAR(both_invalid.w_laser + both_invalid.w_visual, 1.0, 1e-12);
}

TEST(Configuration, RejectsUnsafeThresholds)
{
  lq::QualityConfig config;
  config.failure_recovery_threshold = config.failure_enter_threshold;
  EXPECT_THROW(lq::validate_config(config), std::invalid_argument);
}
