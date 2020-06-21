#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>
// optimizer class, here we use Gauss-Newton
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

class PhotoMetricErrorFactor : public 
        gtsam::NoiseModelFactor2<gtsam::Pose3,
                                 gtsam::Pose3> {

  PhotoMetricErrorFactor() = default;

  PhotoMetricErrorFactor(const gtsam::Key& intrinsics_key,
                         const gtsam::Key& pose_host_key, 
                         const gtsam::Key& pose_target_key, 
                         const gtsam::Key& inverse_depth_key, 
                         const gtsam::Vector& host_pixel, 
                         const double& pixel_intensity,
                         const gtsam::SharedNoiseModel& model) : 
                         gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, intrinsics_key, pose_host_key)
                         , intrinsics_key_(intrinsics_key)
                         , inverse_depth_key_(inverse_depth_key)
                         , target_pose_key_(pose_target_key)
                         , host_pose_key_(pose_host_key) {

  }
  

  gtsam::Vector evaluateError(const gtsam::Pose3& T_host_world, 
                       const gtsam::Pose3& T_target_world,
                       boost::optional<gtsam::Matrix&> H = boost::none) const
  {
    if (H) (*H) = (gtsam::Matrix(2,3)<< 1.0,0.0,0.0, 0.0,1.0,0.0).finished();
    return (gtsam::Vector(2));
  }

  const gtsam::Key intrinsics_key_;
  const gtsam::Key inverse_depth_key_;
  const gtsam::Key target_pose_key_;
  const gtsam::Key host_pose_key_;
};