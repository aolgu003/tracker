#ifndef PHOTOMETRIC_ERROR_FACTOR_H
#define PHOTOMETRIC_ERROR_FACTOR_H

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Pose3.h>

class PhotoMetricErrorFactor : public 
        gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Vector, gtsam::Vector> {
public:

  PhotoMetricErrorFactor() = default;

  PhotoMetricErrorFactor(const gtsam::Key& pose_host_key,
                         const gtsam::Key& pose_target_key, 
                         const gtsam::Key& intrinsics_key, 
                         const gtsam::Key& inverse_depth_key, 
                         const gtsam::Vector& host_pixel, 
                         const cv::Mat& pixel_patch,
                         const gtsam::SharedNoiseModel& model) : 
                         gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Vector, gtsam::Vector>(
                            model, 
                            pose_host_key, 
                            pose_target_key, 
                            intrinsics_key, 
                            inverse_depth_key)
                         , intrinsics_key_(intrinsics_key)
                         , inverse_depth_key_(inverse_depth_key)
                         , target_pose_key_(pose_target_key)
                         , host_pose_key_(pose_host_key) {

  }
  

  gtsam::Vector evaluateError(const gtsam::Pose3& T_host_world, 
                       const gtsam::Pose3& T_target_world,
                       const gtsam::Vector& intrinsics,
                       const gtsam::Vector& depth,
                       boost::optional<gtsam::Matrix&> H1 = boost::none,
                       boost::optional<gtsam::Matrix&> H2 = boost::none,
                       boost::optional<gtsam::Matrix&> H3 = boost::none,
                       boost::optional<gtsam::Matrix&> H4 = boost::none) const  
  {
    auto T_host_target =  T_host_world * .inverse();
    return (gtsam::Vector(1));
  }

  const gtsam::Key intrinsics_key_;
  const gtsam::Key inverse_depth_key_;
  const gtsam::Key target_pose_key_;
  const gtsam::Key host_pose_key_;
};

#endif  PHOTOMETRIC_ERROR_FACTOR