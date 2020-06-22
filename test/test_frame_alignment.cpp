// Bring in gtest
#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "../camera.h"
#include "../photometric_error_factor.h"


class PROJECTION_TEST : public ::testing::Test {
 protected:
  void SetUp() override {
    cam = Camera(1, 1, 10, 10);
  }

  void TearDown() override {}

  Camera cam;
 };

TEST_F(PROJECTION_TEST, compute_center_pixel_bearing) {
  Eigen::Vector3d bearing = cam.ComputePixelBearing(Eigen::Vector2d(10, 10));

  EXPECT_EQ(bearing(0), 0);
  EXPECT_EQ(bearing(1), 0);
  EXPECT_EQ(bearing(2), 1);
}

TEST_F(PROJECTION_TEST, compute_tl_pixel_bearing) {
  Eigen::Vector3d bearing = cam.ComputePixelBearing(Eigen::Vector2d(0, 0));

  EXPECT_LT(bearing(0), 0);
  EXPECT_LT(bearing(1), 0);
  EXPECT_GT(bearing(2), 0);
}

TEST_F(PROJECTION_TEST, compute_tr_pixel_bearing) {
  Eigen::Vector3d bearing = cam.ComputePixelBearing(Eigen::Vector2d(20, 0));

  EXPECT_GT(bearing(0), 0);
  EXPECT_LT(bearing(1), 0);
  EXPECT_GT(bearing(2), 0);
}

TEST_F(PROJECTION_TEST, compute_br_pixel_bearing) {
  Eigen::Vector3d bearing = cam.ComputePixelBearing(Eigen::Vector2d(20, 20));

  EXPECT_GT(bearing(0), 0);
  EXPECT_GT(bearing(1), 0);
  EXPECT_GT(bearing(2), 0);
}

TEST_F(PROJECTION_TEST, projection) {
  Eigen::Vector2d project = cam.ComputePointPixelLocation(Eigen::Vector3d(0, 0, 40));

  EXPECT_EQ(project(0), 10);
  EXPECT_EQ(project(1), 10);
}

class RESIDUAL_TEST : public ::testing::Test {
 protected:
  void SetUp() override {
    cv::Mat image(640, 512, CV_8UC1, cv::Scalar(0));
    cv::Mat patch = (cv::Mat_<double>(5,5) << 
                      60,   90,  90,   90,   60, 
                      60,  170, 190,  170,   60,
                      90,  190, 255,  190,   90,
                      60,  170, 190,  170,   60,
                      60,   90,  90,   90,   60);

    // center at 202
    patch.copyTo(image(cv::Rect(200,200, 5, 5)));
    host_frame = Camera(image, 1, 1, 320, 256);
    auto bearing = host_frame.ComputePixelBearing(Eigen::Vector2d(200,200));
    auto pt = bearing * 5;
    T_target_host = gtsam::Pose3(gtsam::Rot3::ypr(0,0,0), gtsam::Point3(2, 0, 0));
    auto target_pt = T_target_host * pt;
    auto pixel_location = host_frame.ComputePointPixelLocation(target_pt);
    patch.copyTo(image(cv::Rect(static_cast<int>(pixel_location(0)),static_cast<int>(pixel_location(1)), 5, 5)));    
    target_frame = Camera(image, 1, 1, 320, 256);
  }

  void TearDown() override {}
  gtsam::Pose3 T_target_host;
  Camera host_frame;
  Camera target_frame;
};

TEST_F(RESIDUAL_TEST, test_pixel_getter) {
  auto pixel = host_frame.GetPixelValue(Eigen::Vector2d(202, 202));
  EXPECT_EQ(pixel, 255);
}

TEST_F(RESIDUAL_TEST, test_residual) {
  Eigen::Vector2d pixel(202, 202);
  double photometric_error = ComputeResidual(host_frame, target_frame, T_target_host, pixel, 5);
  EXPECT_LT(std::abs(photometric_error), 10);
}

class FACTOR_TEST : public ::testing::Test {
 protected:
  void SetUp() override {
    cv::Mat image(640, 512, CV_8UC1, cv::Scalar(0));
    patch = (cv::Mat_<double>(5,5) << 
                      60,   90,  90,   90,   60, 
                      60,  170, 190,  170,   60,
                      90,  190, 255,  190,   90,
                      60,  170, 190,  170,   60,
                      60,   90,  90,   90,   60);

    // center at 202
    patch.copyTo(image(cv::Rect(200,200, 5, 5)));
    host_frame = Camera(image, 1, 1, 320, 256);
    auto bearing = host_frame.ComputePixelBearing(Eigen::Vector2d(200,200));
    auto pt = bearing * 5;
    T_target_host = gtsam::Pose3(gtsam::Rot3::ypr(0,0,0), gtsam::Point3(2, 0, 0));
    auto target_pt = T_target_host * pt;
    auto pixel_location = host_frame.ComputePointPixelLocation(target_pt);
    patch.copyTo(image(cv::Rect(static_cast<int>(pixel_location(0)),static_cast<int>(pixel_location(1)), 5, 5)));    
    target_frame = Camera(image, 1, 1, 320, 256);
  }

  void TearDown() override {}
  gtsam::Pose3 T_target_host;
  Camera host_frame;
  Camera target_frame;
  cv::Mat patch;
  // Graph
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values graph_values;
  gtsam::Key host_pose_key = 1;
  gtsam::Key target_pose_key = 2;
  gtsam::Key intrinsic_key = 3;
  gtsam::Key depth_one_key = 4;
  gtsam::Key depth_two_key = 5;
  gtsam::Key depth_three_key = 6;
  gtsam::Key depth_four_key = 7;
};

TEST_F(FACTOR_TEST, test_residual) {
  gtsam::NonlinearFactorGraph graph;


  gtsam::Vector pixel(2);
  pixel(0) = 202;
  pixel(1) = 202;

  gtsam::Vector gweight(1);
  const double weight_constant = 2;
  const double grad_mag = 2;
  gweight(0) = weight_constant / (weight_constant + grad_mag); 
  gtsam::noiseModel::Diagonal::shared_ptr photometriceweight =
    gtsam::noiseModel::Diagonal::Sigmas(gweight);

  graph.add(boost::make_shared<PhotoMetricErrorFactor>(host_pose_key,
                                                       target_pose_key,
                                                       intrinsic_key,
                                                       depth_one_key,
                                                       pixel,
                                                       patch,
                                                       photometriceweight));

  graph.add(boost::make_shared<PhotoMetricErrorFactor>(host_pose_key,
                                                       target_pose_key,
                                                       intrinsic_key,
                                                       depth_two_key,
                                                       pixel,
                                                       patch,
                                                       photometriceweight));

  graph.add(boost::make_shared<PhotoMetricErrorFactor>(host_pose_key,
                                                       target_pose_key,
                                                       intrinsic_key,
                                                       depth_three_key,
                                                       pixel,
                                                       patch,
                                                       photometriceweight));

  graph.add(boost::make_shared<PhotoMetricErrorFactor>(host_pose_key,
                                                       target_pose_key,
                                                       intrinsic_key,
                                                       depth_four_key,
                                                       pixel,
                                                       patch,
                                                       photometriceweight));
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
