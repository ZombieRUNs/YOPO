#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gcopter_planner/gcopter_planner.hpp"

#include <gcopter/flatness.hpp>

namespace py = pybind11;

PYBIND11_MODULE(gcopter_planner, m) {
    m.doc() = "GCOPTER trajectory planner — Pybind11 bindings";

    // ------------------------------------------------------------------
    // GcopterPlanner
    // ------------------------------------------------------------------
    py::class_<gcplanner::GcopterPlanner>(m, "GcopterPlanner")
        .def(py::init<const std::string&>(), py::arg("config_path"))

        .def("loadScene",
             &gcplanner::GcopterPlanner::loadScene,
             py::arg("ply_path"),
             py::arg("map_bound"),
             py::arg("voxel_width") = 0.25,
             py::arg("dilate_radius") = 0.5,
             R"(Load PLY point cloud and build dilated VoxelMap.

Args:
    ply_path:      Path to .ply file (PCL PointXYZ, world frame).
    map_bound:     [xmin, xmax, ymin, ymax, zmin, zmax]
    voxel_width:   Voxel side length in metres (default 0.25).
    dilate_radius: Safety margin in metres (default 0.5).
)")

        .def("plan",
             [](gcplanner::GcopterPlanner& planner,
                const Eigen::Vector3d& start_pos,
                const Eigen::Vector3d& start_vel,
                const Eigen::Vector3d& start_acc,
                const Eigen::Vector3d& goal_pos,
                const Eigen::Vector3d& goal_vel,
                const Eigen::Vector3d& goal_acc) -> py::dict {
                 auto r = planner.plan(start_pos, start_vel, start_acc,
                                       goal_pos, goal_vel, goal_acc);
                 py::dict d;
                 d["success"] = r.success;
                 d["total_duration"] = r.total_duration;
                 d["num_pieces"] = r.num_pieces;
                 if (r.success) {
                     d["durations"] = r.durations;
                     // coeffs shape: [N*3, 6]; reshape to [N, 3, 6] in Python.
                     // Coefficient order: col(0)=t^5, ..., col(5)=const.
                     d["coeffs"] = r.coeffs;
                     // waypoints shape: [M, 13]
                     // columns: [t, px,py,pz, vx,vy,vz, ax,ay,az, jx,jy,jz]
                     d["waypoints"] = r.waypoints;
                 }
                 return d;
             },
             py::arg("start_pos"),
             py::arg("start_vel"),
             py::arg("start_acc"),
             py::arg("goal_pos"),
             py::arg("goal_vel"),
             py::arg("goal_acc"),
             R"(Plan a MINCO trajectory from start to goal.

Returns a dict with keys:
    success        (bool)
    total_duration (float, seconds)
    num_pieces     (int)
    durations      (ndarray [N])
    coeffs         (ndarray [N*3, 6], reshape to [N,3,6];
                    descending order: col0=t^5, col5=constant)
    waypoints      (ndarray [M, 13]:
                    [t, px,py,pz, vx,vy,vz, ax,ay,az, jx,jy,jz])
)")

        .def("getConfig",
             [](const gcplanner::GcopterPlanner& p) -> py::dict {
                 const auto& c = p.getConfig();
                 py::dict d;
                 d["weight_t"]      = c.weight_t;
                 d["dt"]            = c.dt;
                 d["physical_params"] = c.physical_params;
                 return d;
             });

    // ------------------------------------------------------------------
    // FlatnessMap — used in Python to convert (vel, acc, jerk, yaw) → quat
    // ------------------------------------------------------------------
    py::class_<flatness::FlatnessMap>(m, "FlatnessMap")
        .def(py::init<>())

        .def("reset",
             &flatness::FlatnessMap::reset,
             py::arg("vehicle_mass"),
             py::arg("gravitational_acceleration"),
             py::arg("horizontal_drag_coeff"),
             py::arg("vertical_drag_coeff"),
             py::arg("parasitic_drag_coeff"),
             py::arg("speed_smooth_factor"),
             "Initialise physical parameters.")

        .def("forward",
             [](flatness::FlatnessMap& fm,
                const Eigen::Vector3d& vel,
                const Eigen::Vector3d& acc,
                const Eigen::Vector3d& jer,
                double psi,
                double dpsi) -> py::tuple {
                 double thr;
                 Eigen::Vector4d quat;
                 Eigen::Vector3d omg;
                 fm.forward(vel, acc, jer, psi, dpsi, thr, quat, omg);
                 return py::make_tuple(thr, quat, omg);
             },
             py::arg("vel"),
             py::arg("acc"),
             py::arg("jer"),
             py::arg("psi"),
             py::arg("dpsi") = 0.0,
             R"(Differential flatness mapping.

Returns:
    (thr, quat, omg)
        thr:  float, thrust magnitude [N]
        quat: ndarray [4], [qw, qx, qy, qz] world frame
        omg:  ndarray [3], body angular velocity [rad/s]
)");
}
