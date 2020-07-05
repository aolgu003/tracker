#ifndef PHOTOMETRIC_ERROR_FACTOR_H
#define PHOTOMETRIC_ERROR_FACTOR_H

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose3.h>
#include <camera.h>

class PhotoMetricErrorFactor : public 
        gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Vector, gtsam::Vector> {
public:

  PhotoMetricErrorFactor() = default;

  PhotoMetricErrorFactor(const cv::Mat& target_image,
			                   const gtsam::Key& pose_host_key,
                         const gtsam::Key& pose_target_key, 
                         const gtsam::Key& intrinsics_key, 
                         const gtsam::Key& inverse_depth_key, 
                         const gtsam::Vector& host_pixel_meas, 
                         const double& host_pixel_value,
                         const gtsam::SharedNoiseModel& model) : 
                         gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Vector, gtsam::Vector>(
                            model, 
                            pose_host_key, 
                            pose_target_key, 
                            intrinsics_key, 
                            inverse_depth_key)
                         , target_image_(target_image)
                         , intrinsics_key_(intrinsics_key)
                         , inverse_depth_key_(inverse_depth_key)
                         , target_pose_key_(pose_target_key)
                         , host_pose_key_(pose_host_key)
                         , pixel_meas_(host_pixel_meas)
                         , meas_value_(host_pixel_value) {}
  
  gtsam::Vector evaluateError(const gtsam::Pose3& T_host_world, 
                       const gtsam::Pose3& T_target_world,
                       const gtsam::Vector& intrinsics,
                       const gtsam::Vector& inv_depth,
                       boost::optional<gtsam::Matrix&> H1 = boost::none,
                       boost::optional<gtsam::Matrix&> H2 = boost::none,
                       boost::optional<gtsam::Matrix&> H3 = boost::none,
                       boost::optional<gtsam::Matrix&> H4 = boost::none) const  
  {
    auto T_target_host = T_target_world * T_host_world.inverse();
    Camera target_frame(target_image_, intrinsics);
    auto host_bearing = target_frame.ComputePixelBearing(pixel_meas_);
    auto host_point = host_bearing * 1/inv_depth(0);
    auto predicted_target_point = T_target_host * host_point;
    auto target_pixel = target_frame.ComputePointPixelLocation(predicted_target_point);
    
    auto target_intensity = target_frame.GetPixelValue(target_pixel);
    gtsam::Vector error(1);
    error(0) = target_intensity - meas_value_;
    return error;
  }

  const cv::Mat& target_image_;
  const double& meas_value_;
  const gtsam::Vector pixel_meas_;
  const gtsam::Key intrinsics_key_;
  const gtsam::Key inverse_depth_key_;
  const gtsam::Key target_pose_key_;
  const gtsam::Key host_pose_key_;
};

#endif  PHOTOMETRIC_ERROR_FACTOR
