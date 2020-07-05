#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <gtsam/geometry/Pose3.h>

class Camera {
public:
  Camera() : 
  fu_(1)
  , fv_(1)
  , pu_(1)
  , pv_(1)
  {}

  Camera(double fu, double fv, double pu, double pv) : 
  fu_(fu)
  , fv_(fv)
  , pu_(pu)
  , pv_(pv) {}

  Camera(const cv::Mat& frame, const gtsam::Vector& intrinsics) :
    Camera(frame, intrinsics(0), intrinsics(1), intrinsics(2), intrinsics(3)) {}

  Camera(const cv::Mat& frame, double fu, double fv, double pu, double pv) : 
  fu_(fu)
  , fv_(fv)
  , pu_(pu)
  , pv_(pv)
  , frame_(frame) {
    int ddepth = CV_16S;
    cv::Sobel(frame_, frame_gx_, ddepth, 1, 0);
    cv::Sobel(frame_, frame_gy_, ddepth, 0, 1);
  }

  Eigen::Vector2d ComputePointPixelLocation(const Eigen::Vector3d& point) const {
    double u = fu_ * point(0) / point(2) + pu_;
    double v = fv_ * point(1) / point(2) + pv_;
  
    Eigen::Vector2d pixel;
    pixel << u, v;
    return pixel;
  }

  Eigen::Vector3d ComputePixelBearing(const Eigen::Vector2d& pixel) const {
    double mx = (pixel(0) - pu_)/fu_;
    double my = (pixel(1) - pv_)/fv_;
    double val = std::pow(std::pow(mx,2) + std::pow(my,2) + 1, .5);

    Eigen::Vector3d bearing;
    bearing << mx/val, my/val, 1/val;
    return bearing;
  }


  Eigen::Matrix<double, 4, 6> ComputePoseJacobian(double feature_distance, double x, double y, double z) const {
    Eigen::Matrix<double, 4, 6> J_pose;
    J_pose << 1/feature_distance, 0, 0,  0, z, -y,
                    0, 1/feature_distance, 0, -z, 0,  z,
                    0, 0, 1/feature_distance,  y, -x,  0,
                    0, 0, 0, 0, 0, 0;
    return J_pose;
  }

  Eigen::Matrix<double, 2,4> ComputeProjectionJacobian(const Eigen::Vector3d& point) const {
    Eigen::Matrix<double, 2,4> J_projection;
    J_projection << fu_* point(2), 0,        -fu_* point(0) / std::pow(point(2), 2), 0,
                    0,      fv_ * 1/point(2), -fv_ * point(1) / std::pow(point(2), 2), 0;
    return J_projection;
  }

  double GetPixelValue(const Eigen::Vector2d pixel) const {
    cv::Mat patch;
    cv::getRectSubPix(frame_, cv::Size(5,5), cv::Point2f(pixel(0), pixel(1)), patch);
    return static_cast<double>(patch.at<uchar>(2,2));
  }

private:
  double fu_, fv_, pu_, pv_;
  cv::Mat frame_, frame_gx_, frame_gy_;
};

double ComputeResidual(const Camera& host_frame, 
                       const Camera& target_frame, 
                       const gtsam::Pose3& T_target_host, 
                       const Eigen::Vector2d& host_pixel_meas,
                       const double& range) {
  auto host_bearing = host_frame.ComputePixelBearing(host_pixel_meas);
  auto host_point = host_bearing * range;
  auto target_point = T_target_host * host_point;
  auto target_pixel = target_frame.ComputePointPixelLocation(target_point);

  auto target_intensity = target_frame.GetPixelValue(target_pixel);
  auto host_intensity = host_frame.GetPixelValue(host_pixel_meas);
  return target_intensity - host_intensity;
}

#endif CAMERA_H
