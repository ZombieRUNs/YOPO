#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace voxel_map {
class VoxelMap;
}

namespace gcplanner {

struct PlanConfig {
    // Optimizer
    double weight_t{20.0};
    double smoothing_eps{1.0e-2};
    int integral_res{16};
    double rel_cost_tol{1.0e-5};
    double rrt_timeout{0.1};
    // Trajectory sampling
    double dt{1.0 / 30.0};
    // magnitudeBounds: [v_max, omg_max, theta_max, thrust_min, thrust_max]
    Eigen::Matrix<double, 5, 1> magnitude_bounds;
    // penaltyWeights: [pos, vel, omg, theta, thrust]
    Eigen::Matrix<double, 5, 1> penalty_weights;
    // physicalParams: [mass, g, horiz_drag, vert_drag, parasitic_drag, speed_eps]
    Eigen::Matrix<double, 6, 1> physical_params;
};

struct PlanResult {
    bool success{false};
    double total_duration{0.0};
    int num_pieces{0};
    // durations: [N]
    Eigen::VectorXd durations;
    // coeffs: [N*3, 6], reshape to [N, 3, 6] in Python.
    // Descending order: col(0)=t^5, col(5)=constant term.
    Eigen::MatrixXd coeffs;
    // waypoints: [M, 13] — [t, px,py,pz, vx,vy,vz, ax,ay,az, jx,jy,jz]
    Eigen::MatrixXd waypoints;
};

class GcopterPlanner {
   public:
    explicit GcopterPlanner(const std::string& config_path);
    ~GcopterPlanner();

    // Load PLY point cloud and build dilated VoxelMap.
    // map_bound: [xmin, xmax, ymin, ymax, zmin, zmax]
    void loadScene(const std::string& ply_path,
                   const std::vector<double>& map_bound,
                   double voxel_width,
                   double dilate_radius);

    PlanResult plan(const Eigen::Vector3d& start_pos,
                    const Eigen::Vector3d& start_vel,
                    const Eigen::Vector3d& start_acc,
                    const Eigen::Vector3d& goal_pos,
                    const Eigen::Vector3d& goal_vel,
                    const Eigen::Vector3d& goal_acc) const;

    const PlanConfig& getConfig() const { return config_; }

   private:
    PlanConfig config_;
    std::unique_ptr<voxel_map::VoxelMap> map_;
    double voxel_width_{0.25};
};

}  // namespace gcplanner
