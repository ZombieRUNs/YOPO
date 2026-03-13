#include "gcopter_planner/gcopter_planner.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gcopter/geo_utils.hpp>
#include <gcopter/gcopter.hpp>
#include <gcopter/sfc_gen.hpp>
#include <gcopter/trajectory.hpp>
#include <gcopter/voxel_map.hpp>

namespace gcplanner {

// ---------------------------------------------------------------------------
// Construction / config loading
// ---------------------------------------------------------------------------

GcopterPlanner::GcopterPlanner(const std::string& config_path) {
    YAML::Node cfg = YAML::LoadFile(config_path);

    config_.weight_t     = cfg["WeightT"].as<double>(20.0);
    config_.smoothing_eps = cfg["SmoothingEps"].as<double>(1.0e-2);
    config_.integral_res  = cfg["IntegralIntervs"].as<int>(16);
    config_.rel_cost_tol  = cfg["RelCostTol"].as<double>(1.0e-5);
    config_.rrt_timeout   = cfg["TimeoutRRT"].as<double>(0.1);
    config_.dt            = cfg["SamplingDT"].as<double>(1.0 / 30.0);

    // magnitudeBounds: [v_max, omg_max, theta_max, thrust_min, thrust_max]
    config_.magnitude_bounds(0) = cfg["MaxVelMag"].as<double>(4.0);
    config_.magnitude_bounds(1) = cfg["MaxBdrMag"].as<double>(2.1);
    config_.magnitude_bounds(2) = cfg["MaxTiltAngle"].as<double>(1.05);
    config_.magnitude_bounds(3) = cfg["MinThrust"].as<double>(2.0);
    config_.magnitude_bounds(4) = cfg["MaxThrust"].as<double>(12.0);

    // penaltyWeights: 5-element list
    auto chi = cfg["ChiVec"].as<std::vector<double>>(
        std::vector<double>{1e4, 1e4, 1e4, 1e4, 1e5});
    for (int i = 0; i < 5; ++i) config_.penalty_weights(i) = chi[i];

    // physicalParams: [mass, g, horiz_drag, vert_drag, parasitic_drag, speed_eps]
    config_.physical_params(0) = cfg["VehicleMass"].as<double>(0.61);
    config_.physical_params(1) = cfg["GravAcc"].as<double>(9.8);
    config_.physical_params(2) = cfg["HorizDrag"].as<double>(0.70);
    config_.physical_params(3) = cfg["VertDrag"].as<double>(0.80);
    config_.physical_params(4) = cfg["ParasDrag"].as<double>(0.01);
    config_.physical_params(5) = cfg["SpeedEps"].as<double>(0.0001);
}

GcopterPlanner::~GcopterPlanner() = default;

// ---------------------------------------------------------------------------
// Scene loading
// ---------------------------------------------------------------------------

void GcopterPlanner::loadScene(const std::string& ply_path,
                                const std::vector<double>& map_bound,
                                double voxel_width,
                                double dilate_radius) {
    if (map_bound.size() != 6)
        throw std::invalid_argument("map_bound must have 6 elements");

    voxel_width_ = voxel_width;

    Eigen::Vector3d origin(map_bound[0], map_bound[2], map_bound[4]);
    Eigen::Vector3d corner(map_bound[1], map_bound[3], map_bound[5]);
    Eigen::Vector3d extent = corner - origin;
    Eigen::Vector3i size =
        (extent / voxel_width).cast<int>() + Eigen::Vector3i::Ones();

    map_ = std::make_unique<voxel_map::VoxelMap>(size, origin, voxel_width);

    // Load PLY and fill VoxelMap
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(ply_path, *cloud) == -1)
        throw std::runtime_error("Failed to load PLY: " + ply_path);

    for (const auto& pt : cloud->points) {
        if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z))
            map_->setOccupied(Eigen::Vector3d(pt.x, pt.y, pt.z));
    }

    // Dilate for safety margin
    int dilate_voxels = static_cast<int>(std::ceil(dilate_radius / voxel_width));
    map_->dilate(dilate_voxels);
}

bool GcopterPlanner::isFree(const Eigen::Vector3d& pos) const {
    if (!map_) {
        return false;
    }
    return !map_->query(pos);
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

PlanResult GcopterPlanner::plan(const Eigen::Vector3d& start_pos,
                                 const Eigen::Vector3d& start_vel,
                                 const Eigen::Vector3d& start_acc,
                                 const Eigen::Vector3d& goal_pos,
                                 const Eigen::Vector3d& goal_vel,
                                 const Eigen::Vector3d& goal_acc) const {
    PlanResult result;

    if (!map_) {
        std::cout << "[gcopter_planner] plan aborted: map not loaded" << std::endl;
        result.success = false;
        return result;
    }

    const bool start_occupied = map_->query(start_pos);
    const bool goal_occupied = map_->query(goal_pos);
    std::cout << "[gcopter_planner] plan begin"
              << " start=[" << start_pos.transpose() << "]"
              << " goal=[" << goal_pos.transpose() << "]"
              << " start_occ=" << static_cast<int>(start_occupied)
              << " goal_occ=" << static_cast<int>(goal_occupied)
              << std::endl;

    // Skip invalid endpoint pairs early. This avoids feeding OMPL with invalid
    // starts/goals and makes failure reason explicit in logs.
    if (start_occupied || goal_occupied) {
        std::cout << "[gcopter_planner] endpoint invalid -> skip"
                  << " start_occ=" << static_cast<int>(start_occupied)
                  << " goal_occ=" << static_cast<int>(goal_occupied)
                  << std::endl;
        return result;
    }

    // 1. RRT* path planning
    std::vector<Eigen::Vector3d> route;
    const auto t0 = std::chrono::steady_clock::now();
    double path_cost = sfc_gen::planPath<voxel_map::VoxelMap>(
        start_pos, goal_pos,
        map_->getOrigin(), map_->getCorner(),
        map_.get(), config_.rrt_timeout,
        route);
    const auto t1 = std::chrono::steady_clock::now();
    const auto rrt_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // NOTE:
    // In Release (-Ofast), finite-math assumptions may make isinf/isnan checks
    // unreliable. Use route cardinality as hard success criterion.
    const bool rrt_failed = route.size() < 2;
    if (rrt_failed) {
        std::cout << "[gcopter_planner] RRT failed"
                  << " elapsed_ms=" << rrt_ms
                  << " route_pts=" << route.size()
                  << " path_cost=" << path_cost
                  << " timeout_s=" << config_.rrt_timeout << std::endl;
        return result;
    }

    std::cout << "[gcopter_planner] RRT success"
              << " elapsed_ms=" << rrt_ms
              << " route_pts=" << route.size()
              << " path_cost=" << path_cost << std::endl;

    // 2. Surface points for corridor generation
    std::vector<Eigen::Vector3d> surf_pts;
    map_->getSurf(surf_pts);

    // 3. Convex corridor (FIRI)
    std::vector<Eigen::MatrixX4d> h_polys;
    const auto t2 = std::chrono::steady_clock::now();
    sfc_gen::convexCover(route, surf_pts,
                          map_->getOrigin(), map_->getCorner(),
                          7.0, 3.0, h_polys);
    sfc_gen::shortCut(h_polys);
    const auto t3 = std::chrono::steady_clock::now();
    const auto corridor_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::cout << "[gcopter_planner] corridor built"
              << " elapsed_ms=" << corridor_ms
              << " surf_pts=" << surf_pts.size()
              << " corridors=" << h_polys.size() << std::endl;

    if (h_polys.empty()) {
        std::cout << "[gcopter_planner] corridor empty -> skip optimize" << std::endl;
        return result;
    }

    // 4. Boundary conditions
    Eigen::Matrix3d ini_state, fin_state;
    ini_state.col(0) = start_pos;
    ini_state.col(1) = start_vel;
    ini_state.col(2) = start_acc;
    fin_state.col(0) = goal_pos;
    fin_state.col(1) = goal_vel;
    fin_state.col(2) = goal_acc;

    // 5. GCOPTER setup
    gcopter::GCOPTER_PolytopeSFC optimizer;
    Eigen::VectorXd mag_bounds = config_.magnitude_bounds;
    Eigen::VectorXd pen_weights = config_.penalty_weights;
    Eigen::VectorXd phy_params = config_.physical_params;

    if (!optimizer.setup(config_.weight_t,
                          ini_state, fin_state,
                          h_polys, INFINITY,
                          config_.smoothing_eps,
                          config_.integral_res,
                          mag_bounds, pen_weights, phy_params))
    {
        std::cout << "[gcopter_planner] optimizer setup failed"
                  << " corridors=" << h_polys.size() << std::endl;
        return result;
    }

    // 6. Optimize
    Trajectory<5> traj;
    const auto t4 = std::chrono::steady_clock::now();
    const double opt_cost = optimizer.optimize(traj, config_.rel_cost_tol);
    const auto t5 = std::chrono::steady_clock::now();
    const auto opt_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();

    // NOTE:
    // flightlib Release uses -Ofast; finite-math assumptions can make isinf/isnan
    // checks unreliable. Use trajectory validity as the hard failure criterion.
    const int piece_num = traj.getPieceNum();
    const double total_T = traj.getTotalDuration();
    const bool invalid_traj = (piece_num <= 0) || (total_T <= 0.0);
    if (invalid_traj) {
        std::cout << "[gcopter_planner] optimize failed"
                  << " elapsed_ms=" << opt_ms
                  << " rel_cost_tol=" << config_.rel_cost_tol
                  << " integral_res=" << config_.integral_res
                  << " cost=" << opt_cost
                  << " pieces=" << piece_num
                  << " total_T=" << total_T
                  << std::endl;
        return result;
    }

    // 7. Extract results
    const int N = piece_num;
    const double T = total_T;
    std::cout << "[gcopter_planner] optimize success"
              << " elapsed_ms=" << opt_ms
              << " cost=" << opt_cost
              << " pieces=" << N
              << " total_T=" << T << std::endl;

    result.success = true;
    result.total_duration = T;
    result.num_pieces = N;

    // Durations and coefficients (descending order: col(0)=t^5, col(5)=const)
    result.durations.resize(N);
    result.coeffs.resize(N * 3, 6);
    for (int i = 0; i < N; ++i) {
        result.durations(i) = traj[i].getDuration();
        auto cmat = traj[i].getCoeffMat();  // Matrix<double, 3, 6>
        result.coeffs.block<3, 6>(i * 3, 0) = cmat;
    }

    // Waypoints sampled at dt
    const int M = std::max(1, static_cast<int>(std::floor(T / config_.dt)));
    result.waypoints.resize(M, 13);
    for (int k = 0; k < M; ++k) {
        double t = k * config_.dt;
        Eigen::Vector3d pos = traj.getPos(t);
        Eigen::Vector3d vel = traj.getVel(t);
        Eigen::Vector3d acc = traj.getAcc(t);
        Eigen::Vector3d jer = traj.getJer(t);
        result.waypoints.row(k) << t,
            pos(0), pos(1), pos(2),
            vel(0), vel(1), vel(2),
            acc(0), acc(1), acc(2),
            jer(0), jer(1), jer(2);
    }

    return result;
}

}  // namespace gcplanner
