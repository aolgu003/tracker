// Bring in gtest
#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "camera.h"
#include "photometric_error_factor.h"


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
    host_image = cv::Mat(640, 512, CV_8UC1, cv::Scalar(0));
    
    patch = (cv::Mat_<double>(5,5) << 
                      60,   90,  90,   90,   60, 
                      60,  170, 190,  170,   60,
                      90,  190, 255,  190,   90,
                      60,  170, 190,  170,   60,
                      60,   90,  90,   90,   60);
    // center at 202
    patch.copyTo(host_image(cv::Rect(200,200, 5, 5)));
    patch.copyTo(host_image(cv::Rect(300,200, 5, 5)));
    patch.copyTo(host_image(cv::Rect(200,300, 5, 5)));

    
    host_frame = Camera(host_image, 1, 1, 320, 256);

    auto p0_pixel_location = Eigen::Vector2d(200,200);
    auto p1_pixel_location = Eigen::Vector2d(204,200);
    auto p2_pixel_location = Eigen::Vector2d(204,204);
    auto p3_pixel_location = Eigen::Vector2d(200,204);

    auto bearing_p0 = host_frame.ComputePixelBearing(p0_pixel_location);
    auto bearing_p1 = host_frame.ComputePixelBearing(p1_pixel_location);
    auto bearing_p2 = host_frame.ComputePixelBearing(p2_pixel_location);
    auto bearing_p3 = host_frame.ComputePixelBearing(p3_pixel_location);

    auto pt0 = bearing_p0 * depth;
    auto pt1 = bearing_p1 * depth;
    auto pt2 = bearing_p2 * depth;
    auto pt3 = bearing_p3 * depth;

    T_p0_world = gtsam::Pose3(gtsam::Rot3::ypr(0,0,0), gtsam::Point3(2, 0, 0));
    T_p1_world = gtsam::Pose3(gtsam::Rot3::ypr(0,0,0), gtsam::Point3(3, 0, 0));
    
    auto p1_pt0 = T_p1_world * T_p0_world.inverse() * pt0;
    auto p1_pt1 = T_p1_world * T_p0_world.inverse() * pt1;
    auto p1_pt2 = T_p1_world * T_p0_world.inverse() * pt2;
    auto p1_pt3 = T_p1_world * T_p0_world.inverse() * pt3;
 
    auto pt0_target_pixel = host_frame.ComputePointPixelLocation(p1_pt0);
    auto pt1_target_pixel = host_frame.ComputePointPixelLocation(p1_pt1);
    auto pt2_target_pixel = host_frame.ComputePointPixelLocation(p1_pt2);
    auto pt3_target_pixel = host_frame.ComputePointPixelLocation(p1_pt3);

    std::vector<cv::Point2f> host_pixel_locations(4);
    host_pixel_locations[0] = cv::Point2f(p0_pixel_location(0), p0_pixel_location(1));
    host_pixel_locations[1] = cv::Point2f(p1_pixel_location(0), p1_pixel_location(1));
    host_pixel_locations[2] = cv::Point2f(p2_pixel_location(0), p2_pixel_location(1));
    host_pixel_locations[3] = cv::Point2f(p3_pixel_location(0), p3_pixel_location(1));
    
    std::vector<cv::Point2f> target_pixel_locations(4);
    target_pixel_locations[0] = cv::Point2f(pt0_target_pixel(0), pt0_target_pixel(1));
    target_pixel_locations[1] = cv::Point2f(pt1_target_pixel(0), pt1_target_pixel(1));
    target_pixel_locations[2] = cv::Point2f(pt2_target_pixel(0), pt2_target_pixel(1));
    target_pixel_locations[3] = cv::Point2f(pt3_target_pixel(0), pt3_target_pixel(1));
    cv::Mat h_target_host = cv::getPerspectiveTransform(host_pixel_locations, target_pixel_locations);

    cv::warpPerspective(host_image, target_image, h_target_host, host_image.size());
    target_frame = Camera(target_image, 1, 1, 320, 256);

    std::cout << p0_pixel_location << std::endl;
    std::cout << p1_pixel_location << std::endl;

    auto patch_center_bearing = target_frame.ComputePixelBearing(Eigen::Vector2d(202, 202));
  }

  void TearDown() override {}
  gtsam::Pose3 T_p0_world;
  gtsam::Pose3 T_p1_world;


  Camera host_frame;
  double depth = 2;
  
  Camera target_frame;

  cv::Mat patch;
  cv::Mat host_image;
  cv::Mat target_image;
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

  graph.add(boost::make_shared<PhotoMetricErrorFactor>(target_image,
                                                        host_pose_key,
                                                        target_pose_key,
                                                        intrinsic_key,
                                                        depth_one_key,
                                                        pixel,
                                                        255,
                                                        photometriceweight));

  pixel(0) = 302;
  pixel(1) = 202;
  graph.add(boost::make_shared<PhotoMetricErrorFactor>(target_image,
                                                        host_pose_key,
                                                        target_pose_key,
                                                        intrinsic_key,
                                                        depth_two_key,
                                                        pixel,
                                                        255,
                                                        photometriceweight));

  pixel(0) = 202;
  pixel(1) = 302;
  graph.add(boost::make_shared<PhotoMetricErrorFactor>(target_image,
                                                        host_pose_key,
                                                        target_pose_key,
                                                        intrinsic_key,
                                                        depth_three_key,
                                                        pixel,
                                                        255,
                                                        photometriceweight));
  gtsam::Vector intrinsics(4);
  intrinsics(0) = 1;
  intrinsics(1) = 1;
  intrinsics(2) = 320;
  intrinsics(3) = 256;
  
  gtsam::Values initial_values;
  initial_values.insert(host_pose_key, T_p0_world);
  initial_values.insert(target_pose_key, T_p1_world);
  initial_values.insert(intrinsic_key, intrinsics);

  gtsam::Vector invdepth(1);
  invdepth(0) = 1/depth;
  initial_values.insert(depth_one_key, invdepth);
  initial_values.insert(depth_two_key, invdepth);
  initial_values.insert(depth_three_key, invdepth);

  gtsam::Values results = gtsam::LevenbergMarquardtOptimizer(graph, initial_values).optimize();
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
