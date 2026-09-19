#include "localization_quality/quality_core.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <limits>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif
#ifndef BOOTSTRAP_MODEL_PATH
#define BOOTSTRAP_MODEL_PATH ""
#endif
#ifndef FORMAL_MODEL_PATH
#define FORMAL_MODEL_PATH ""
#endif

namespace lq = localization_quality;

TEST(AdvancedVisualQuality, ExposureEdgesAndTemporalFeaturesAreBounded)
{
  lq::QualityConfig config;
  cv::Mat checker(160, 160, CV_8UC1);
  for (int row = 0; row < checker.rows; ++row) {
    for (int column = 0; column < checker.cols; ++column) {
      checker.at<unsigned char>(row, column) =
        ((row / 10 + column / 10) % 2) ? 220 : 35;
    }
  }
  lq::VisualQualityEvaluator evaluator(config);
  const lq::VisualQuality first = evaluator.evaluate(checker);
  const lq::VisualQuality second = evaluator.evaluate(checker);
  EXPECT_TRUE(first.valid);
  EXPECT_GT(first.edge_density, 0.0);
  EXPECT_GT(first.trackable_features, 0);
  EXPECT_GE(second.feature_match_rate, 0.0);
  EXPECT_LE(second.feature_match_rate, 1.0);
  EXPECT_GE(second.quality, 0.0);
  EXPECT_LE(second.quality, 1.0);

  const lq::VisualQuality lost =
    lq::apply_visual_localization_status(second, "lost", 1.0);
  EXPECT_FALSE(lost.tracking_ok);
  EXPECT_LT(lost.quality, second.quality);
}

TEST(AdvancedVisualQuality, CompressedImageHandlesValidAndInvalidPayloads)
{
  lq::QualityConfig config;
  cv::Mat image(64, 64, CV_8UC3, cv::Scalar(80, 140, 200));
  cv::line(image, cv::Point(0, 0), cv::Point(63, 63), cv::Scalar(255, 255, 255), 3);
  std::vector<unsigned char> encoded;
  ASSERT_TRUE(cv::imencode(".jpg", image, encoded));
  const lq::VisualQuality valid =
    lq::evaluate_compressed_visual(encoded, config);
  EXPECT_TRUE(valid.valid);
  const lq::VisualQuality invalid =
    lq::evaluate_compressed_visual(std::vector<std::uint8_t>{1, 2, 3}, config);
  EXPECT_FALSE(invalid.valid);
  EXPECT_EQ(invalid.status, "compressed_decode_error");
}

TEST(AdvancedLaserQuality, ScanGeometryAndPointCloudHandleInvalidData)
{
  lq::QualityConfig config;
  std::vector<float> scan(360, 5.0F);
  for (std::size_t index = 0; index < scan.size(); index += 17) {
    scan[index] = std::numeric_limits<float>::quiet_NaN();
  }
  const lq::LaserQuality laser = lq::evaluate_laser(
    scan, 0.1, 10.0, -3.141592653589793,
    2.0 * 3.141592653589793 / 360.0, config);
  EXPECT_TRUE(laser.valid);
  EXPECT_GE(laser.direction_entropy, 0.0);
  EXPECT_LE(laser.direction_entropy, 1.0);
  EXPECT_GE(laser.geometry_complexity, 0.0);
  EXPECT_LE(laser.geometry_complexity, 1.0);

  std::vector<cv::Point3f> cloud;
  for (int x = -4; x <= 4; ++x) {
    for (int y = -4; y <= 4; ++y) {
      for (int z = -2; z <= 2; ++z) {
        cloud.emplace_back(0.3F * x, 0.3F * y, 0.2F * z);
      }
    }
  }
  cloud.emplace_back(
    std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F);
  const lq::LaserQuality point_quality =
    lq::evaluate_point_cloud(cloud, config);
  EXPECT_TRUE(point_quality.valid);
  EXPECT_GT(point_quality.spatial_coverage, 0.0);
  EXPECT_GT(point_quality.pca_eigenvalues[0], 0.0);
  EXPECT_NEAR(
    point_quality.linearity + point_quality.planarity +
    point_quality.scattering, 1.0, 1e-6);
  const lq::LaserQuality sparse = lq::evaluate_point_cloud(
    std::vector<cv::Point3f>{{1.0F, 0.0F, 0.0F}}, config);
  EXPECT_FALSE(sparse.valid);
  EXPECT_EQ(sparse.status, "insufficient_points");
}

TEST(ContextModel, LoadsAndChangesWeightsWithContext)
{
  lq::ContextMlpModel model;
  std::string error;
  ASSERT_TRUE(model.load(
    std::string(TEST_DATA_DIR) + "/test_context_mlp.yaml", &error)) << error;

  lq::EnvironmentContext bright;
  bright.scene_probabilities = {{0.2, 0.3, 0.5}};
  bright.illumination_quality = 1.0;
  bright.geometry_complexity = 0.0;
  bright.openness = 0.5;
  const auto bright_weights = model.predict_weights(0.7, 0.7, bright);

  lq::EnvironmentContext geometric = bright;
  geometric.illumination_quality = 0.0;
  geometric.geometry_complexity = 1.0;
  const auto geometric_weights = model.predict_weights(0.7, 0.7, geometric);
  EXPECT_GT(bright_weights[0], geometric_weights[0]);
  EXPECT_NEAR(bright_weights[0] + bright_weights[1], 1.0, 1e-12);
  EXPECT_NEAR(geometric_weights[0] + geometric_weights[1], 1.0, 1e-12);
}

TEST(EnvironmentContext, ProbabilitiesLightingAndGeometryAreExplicit)
{
  lq::LaserQuality corridor;
  corridor.valid = true;
  corridor.corridor_degeneracy = 0.95;
  corridor.directional_anisotropy = 0.90;
  corridor.openness = 0.20;
  corridor.geometry_complexity = 0.15;
  corridor.coverage_rate = 0.9;
  lq::VisualQuality dark;
  dark.valid = true;
  dark.illumination_level = 0.10;
  dark.illumination_quality = 0.20;
  dark.illumination_state = "dark";
  const lq::EnvironmentContext context =
    lq::evaluate_environment_context(corridor, dark);
  EXPECT_TRUE(context.valid);
  EXPECT_EQ(context.scene, lq::SceneType::corridor);
  EXPECT_EQ(context.illumination_state, "dark");
  EXPECT_DOUBLE_EQ(context.geometry_complexity, 0.15);
  EXPECT_NEAR(
    context.scene_probabilities[0] +
    context.scene_probabilities[1] +
    context.scene_probabilities[2], 1.0, 1e-12);
}

TEST(ContextModel, NumPyExportedBootstrapModelLoadsInCpp)
{
  lq::ContextMlpModel model;
  std::string error;
  ASSERT_TRUE(model.load(BOOTSTRAP_MODEL_PATH, &error)) << error;
  lq::VisualQuality visual;
  visual.valid = true;
  visual.quality = 0.7;
  visual.illumination_quality = 0.8;
  visual.edge_density = 0.1;
  visual.trackable_features = 300;
  visual.feature_match_rate = 0.7;
  lq::LaserQuality laser;
  laser.valid = true;
  laser.quality = 0.8;
  laser.valid_rate = 0.95;
  laser.coverage_rate = 0.9;
  laser.direction_entropy = 0.6;
  laser.geometry_complexity = 0.5;
  laser.openness = 0.6;
  EXPECT_GE(model.calibrate_visual(visual), 0.0);
  EXPECT_LE(model.calibrate_visual(visual), 1.0);
  EXPECT_GE(model.calibrate_laser(laser), 0.0);
  EXPECT_LE(model.calibrate_laser(laser), 1.0);
  const lq::EnvironmentContext context =
    model.evaluate_context(laser, visual);
  const auto weights = model.predict_weights(0.8, 0.7, context);
  EXPECT_NEAR(weights[0] + weights[1], 1.0, 1e-12);
}

TEST(ContextModel, FormalNineFeatureCalibrationUsesTrackingStatus)
{
  lq::ContextMlpModel model;
  std::string error;
  ASSERT_TRUE(model.load(FORMAL_MODEL_PATH, &error)) << error;
  lq::VisualQuality tracked;
  tracked.valid = true;
  tracked.quality = 0.10;
  tracked.illumination_quality = 0.20;
  tracked.underexposed_ratio = 0.80;
  tracked.tracking_ok = true;
  tracked.localization_covariance = 0.01;
  lq::VisualQuality lost = tracked;
  lost.tracking_ok = false;
  lost.localization_covariance = 3.0;
  const double tracked_quality = model.calibrate_visual(tracked);
  const double lost_quality = model.calibrate_visual(lost);
  EXPECT_GE(tracked_quality, 0.0);
  EXPECT_LE(tracked_quality, 1.0);
  EXPECT_GE(lost_quality, 0.0);
  EXPECT_LE(lost_quality, 1.0);
  EXPECT_GT(tracked_quality, lost_quality);
}

TEST(Supervision, LabelsAndOracleWeightsAreNormalized)
{
  EXPECT_NEAR(lq::supervised_quality_label(0.0, 0.0), 1.0, 1e-12);
  EXPECT_LT(lq::supervised_quality_label(1.0, 1.0), 0.1);
  const auto weights = lq::oracle_weights(0.1, 0.05, 1.0, 0.5);
  EXPECT_GT(weights[0], weights[1]);
  EXPECT_NEAR(weights[0] + weights[1], 1.0, 1e-12);
}

TEST(QualityGrid, MeanVarianceConfidenceAndUnknownAreCorrect)
{
  lq::QualityGridConfig config;
  config.resolution = 1.0;
  config.width = 4;
  config.height = 3;
  config.origin_x = -1.0;
  config.origin_y = -1.0;
  lq::QualityGrid grid(config);
  EXPECT_TRUE(grid.update(-0.5, -0.5, 0.2, 1.0));
  EXPECT_TRUE(grid.update(-0.5, -0.5, 0.8, 2.0));
  EXPECT_FALSE(grid.update(10.0, 10.0, 0.5, 3.0));
  const lq::QualityGridCell cell = grid.cell(0, 0);
  EXPECT_EQ(cell.count, 2U);
  EXPECT_NEAR(cell.mean, 0.5, 1e-12);
  EXPECT_NEAR(cell.variance, 0.18, 1e-12);
  EXPECT_GT(cell.confidence, 0.0);
  EXPECT_DOUBLE_EQ(cell.last_update, 2.0);
  const std::vector<std::int8_t> occupancy = grid.occupancy_data();
  EXPECT_EQ(occupancy[0], 50);
  EXPECT_EQ(occupancy[1], -1);
}

TEST(SafetyFusion, MissingModelUsesExplicitFallback)
{
  lq::QualityConfig config;
  config.quality_ema_alpha = 1.0;
  config.context_model_path = "/definitely/missing/model.yaml";
  lq::AdaptiveFusion fusion(config);
  lq::LaserQuality laser;
  laser.valid = true;
  laser.quality = 0.8;
  lq::VisualQuality visual;
  visual.valid = true;
  visual.quality = 0.7;
  const lq::FusionResult result = fusion.update(laser, visual);
  EXPECT_EQ(result.strategy, "rule_fallback");
  EXPECT_EQ(result.state, "model_unavailable");
  EXPECT_FALSE(result.model_loaded);
}

TEST(SafetyFusion, HysteresisRecoveryAndInvalidInputAreImmediate)
{
  lq::QualityConfig config;
  config.quality_ema_alpha = 1.0;
  config.fusion_strategy = "rule";
  lq::AdaptiveFusion fusion(config);
  lq::LaserQuality laser;
  laser.valid = true;
  laser.quality = 0.8;
  lq::VisualQuality visual;
  visual.valid = true;
  visual.quality = 0.8;
  EXPECT_EQ(fusion.update(laser, visual).state, "normal");

  laser.quality = 0.2;
  EXPECT_EQ(fusion.update(laser, visual).state, "laser_degraded");
  laser.quality = 0.4;
  EXPECT_EQ(fusion.update(laser, visual).state, "laser_degraded");
  laser.quality = 0.5;
  EXPECT_EQ(fusion.update(laser, visual).state, "normal");

  lq::QualityConfig slow_config = config;
  slow_config.quality_ema_alpha = 0.05;
  lq::AdaptiveFusion slow_fusion(slow_config);
  laser.quality = 0.9;
  ASSERT_EQ(slow_fusion.update(laser, visual).state, "normal");
  laser.valid = false;
  const auto invalid = slow_fusion.update(laser, visual);
  EXPECT_EQ(invalid.state, "laser_invalid");
  EXPECT_DOUBLE_EQ(invalid.q_laser, 0.0);
  EXPECT_NEAR(invalid.w_visual, 0.9, 1e-12);
}

TEST(SafetyFusion, FormalModelMarksSafetyOverrideAndBothDegraded)
{
  lq::QualityConfig config;
  config.quality_ema_alpha = 1.0;
  config.context_model_path = FORMAL_MODEL_PATH;
  lq::AdaptiveFusion fusion(config);
  lq::LaserQuality laser;
  laser.valid = true;
  laser.quality = 0.8;
  laser.valid_rate = 0.9;
  laser.coverage_rate = 0.9;
  lq::VisualQuality visual;
  visual.valid = true;
  visual.quality = 0.1;
  visual.tracking_ok = false;
  visual.localization_covariance = 3.0;
  const auto override = fusion.update(laser, visual);
  EXPECT_EQ(override.strategy, "mlp_safety_override");
  EXPECT_NEAR(override.w_laser, 0.9, 1e-12);

  laser.quality = 0.1;
  const auto both = fusion.update(laser, visual);
  EXPECT_EQ(both.strategy, "both_degraded");
  EXPECT_NEAR(both.w_laser, 0.5, 1e-12);
  EXPECT_NEAR(both.w_visual, 0.5, 1e-12);
}

TEST(SafetyFusion, HardwareDomainGuardRecoversHealthyVisualEvidence)
{
  lq::QualityConfig config;
  config.quality_ema_alpha = 1.0;
  config.context_model_path = FORMAL_MODEL_PATH;
  lq::AdaptiveFusion fusion(config);
  lq::LaserQuality laser;
  laser.valid = true;
  laser.quality = 0.8;
  lq::VisualQuality visual;
  visual.valid = true;
  visual.quality = 0.716;
  visual.illumination_quality = 0.7;
  visual.edge_density = 0.05;
  visual.trackable_features = 250;
  visual.feature_match_rate = 0.5;
  // The real smoke bag has no visual-localization stream.  Unknown tracking
  // is therefore not a positive lost-tracking observation.
  visual.tracking_ok = true;
  visual.localization_covariance = std::numeric_limits<double>::infinity();

  const auto guarded = fusion.update(laser, visual);
  EXPECT_TRUE(guarded.calibration_domain_guard_applied);
  EXPECT_EQ(guarded.strategy, "domain_guard_rule_fallback");
  EXPECT_EQ(guarded.state, "normal");
  EXPECT_FALSE(guarded.visual_degraded);
  EXPECT_NEAR(guarded.q_visual, visual.quality, 1e-12);
  EXPECT_NEAR(guarded.q_laser, laser.quality, 1e-12);

  visual.tracking_ok = false;
  visual.tracking_state = "lost";
  visual.localization_covariance = 3.0;
  lq::AdaptiveFusion lost_fusion(config);
  const auto lost = lost_fusion.update(laser, visual);
  EXPECT_FALSE(lost.calibration_domain_guard_applied);
  EXPECT_TRUE(lost.visual_degraded);
  EXPECT_EQ(lost.strategy, "mlp_safety_override");
}
