"""
Collect GCOPTER reference trajectory dataset using Flightmare scenes.

For each scene:
  1. Spawn random trees and export PLY point cloud.
  2. Load PLY into GCOPTER VoxelMap.
  3. Sample random start/goal pairs (collision-free via Flightmare reset).
  4. Plan MINCO trajectory with GCOPTER (C++ via Pybind11).
  5. Walk the trajectory at 30 Hz, set drone state, collect depth images.
  6. Save coeffs, durations, waypoints, depths, and meta info.

Output layout:
  dataset/gcopter_trajs/
    index.json
    scene_000/
      pointcloud.ply
      traj_0000/
        meta.json
        coeffs.npy       [N*3, 6]  float64 (descending: col0=t^5, col5=const)
        durations.npy    [N]       float64
        waypoints.npy    [M, 13]   float32  [t,px,py,pz,vx,vy,vz,ax,ay,az,jx,jy,jz]
        depths.npy       [M,96,160] float16  (0..1 normalised, 20m clip)
        start_pva.npy    [3, 3]
        goal_pva.npy     [3, 3]
      traj_0001/
        ...
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
from ruamel.yaml import YAML, RoundTripDumper, dump

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SCENES = 50
TRAJS_PER_SCENE = 30
MIN_DIST = 5.0       # metres
MAX_DIST = 20.0      # metres
DT = 1.0 / 30.0      # 30 Hz
SAVE_ROOT = "dataset/gcopter_trajs"

MAP_BOUND = [-25.0, 25.0, -25.0, 25.0, 0.0, 5.0]  # [xmin,xmax,ymin,ymax,zmin,zmax]
VOXEL_WIDTH = 0.25   # metres
DILATE_R = 0.5       # metres

# FLIGHTMARE_PATH defaults to the YOPO root (parent of this script's directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLIGHTMARE_PATH = os.environ.get("FLIGHTMARE_PATH",
                                  os.path.dirname(_SCRIPT_DIR))
PLY_DIR = os.path.join(FLIGHTMARE_PATH, "run", "yopo_sim")
CFG_PATH = os.path.join(FLIGHTMARE_PATH, "flightlib", "configs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quat_to_rot(q):
    """[qw, qx, qy, qz] → 3×3 rotation matrix (world ← body)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def obs_to_world_pva(obs):
    """Convert Flightmare obs (body-frame vel/acc) to world-frame PVA."""
    pos = obs[0, 0:3].copy()
    quat = obs[0, 9:13]          # [qw, qx, qy, qz]
    R = quat_to_rot(quat)        # world ← body
    vel = R @ obs[0, 3:6]
    acc = R @ obs[0, 6:9]
    return pos, vel, acc


def sample_pair(env, planner, min_d, max_d, max_tries=300):
    """Sample a start/goal pair that is free in both Flightmare and GCOPTER map."""
    for _ in range(max_tries):
        obs_s = env.reset()
        sp, sv, sa = obs_to_world_pva(obs_s)
        obs_g = env.reset()
        gp, gv, ga = obs_to_world_pva(obs_g)
        # Enforce non-overlap with point cloud / dilated occupancy before planning.
        if (not planner.isFree(sp)) or (not planner.isFree(gp)):
            continue
        if min_d < np.linalg.norm(gp - sp) < max_d:
            # Use zero boundary conditions for clean start/stop
            return (sp, np.zeros(3), np.zeros(3),
                    gp, np.zeros(3), np.zeros(3))
    return None


def collect_depths(env, flatmap, waypoints):
    """
    Walk the trajectory and collect depth images at each waypoint.

    Args:
        env:        Flightmare FlightEnvVec
        flatmap:    gcopter_planner.FlatnessMap (pre-initialised)
        waypoints:  ndarray [M, 13]

    Returns:
        depths: ndarray [M, 96, 160] float16, normalised to [0, 1]
                (normalisation is done by vec_env_wrapper.getDepthImage)
    """
    M = len(waypoints)
    depths = np.zeros((M, 96, 160), dtype=np.float16)

    for k in range(M):
        t, px, py, pz, vx, vy, vz, ax, ay, az, jx, jy, jz = waypoints[k]
        vel = np.array([vx, vy, vz])
        acc = np.array([ax, ay, az])
        jer = np.array([jx, jy, jz])

        # Yaw = direction of flight; identity quat when near-hover
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed < 1e-3:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            psi = math.atan2(vy, vx)
            _thr, quat, _omg = flatmap.forward(vel, acc, jer, psi, 0.0)
            quat = np.array(quat)

        # setState sets C++ state; render() syncs with Unity and fills image buffer
        env.setState(np.array([px, py, pz]), vel, acc, quat)
        env.render()

        # getDepthImage() returns [num_envs, 1, net_h, net_w], already normalised [0,1]
        depth = env.getDepthImage()[0][0]   # → [96, 160]
        depths[k] = depth.astype(np.float16)

    return depths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    import gcopter_planner
    from flightgym import QuadrotorEnv_v1
    from flightpolicy.envs import vec_env_wrapper as wrapper

    # Flightmare environment — QuadrotorEnv_v1 expects ruamel.yaml-dumped string
    vec_cfg_path = os.path.join(CFG_PATH, "vec_env.yaml")
    cfg = YAML().load(open(vec_cfg_path, "r"))
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = True
    cfg["env"]["supervised"] = False
    cfg["env"]["imitation"] = False
    os.system(FLIGHTMARE_PATH + "/flightrender/RPG_Flightmare/flightmare.x86_64 &")
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env.connectUnity()

    # Use Flightmare world box as GCOPTER map bound
    # env.world_box = [x_min, y_min, z_min, x_max, y_max, z_max]
    wb = env.world_box
    map_bound = [float(wb[0]), float(wb[3]),   # xmin, xmax
                 float(wb[1]), float(wb[4]),   # ymin, ymax
                 max(0.0, float(wb[2])),        # zmin (floor at 0)
                 min(8.0, float(wb[5]))]        # zmax (cap at 8 m)
    print(f"[INFO] world_box={wb}  map_bound={map_bound}", flush=True)

    # GCOPTER planner
    planner_cfg = os.path.join(CFG_PATH, "gcopter_config.yaml")
    planner = gcopter_planner.GcopterPlanner(planner_cfg)

    # FlatnessMap (same physical params as planner config)
    cfg = planner.getConfig()
    pp = cfg["physical_params"]
    flatmap = gcopter_planner.FlatnessMap()
    flatmap.reset(pp[0], pp[1], pp[2], pp[3], pp[4], pp[5])

    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    index = {
        "total_scenes": args.num_scenes,
        "total_trajectories": 0,
        "dt": DT,
        "img_shape": [96, 160],
        "img_range": "0-1 normalised, 20m clip",
        "scenes": [],
    }

    for scene_id in range(args.num_scenes):
        # Generate scene + export PLY
        print(f"[scene {scene_id}] spawnTrees", flush=True)
        env.setMapID(np.array([scene_id]))
        env.spawnTreesAndSavePointcloud(scene_id, spacing=4.0)
        ply_src = os.path.join(PLY_DIR, f"pointcloud-{scene_id}.ply")

        # Load into GCOPTER VoxelMap
        print(f"[scene {scene_id}] loadScene", flush=True)
        planner.loadScene(ply_src, map_bound, VOXEL_WIDTH, DILATE_R)
        print(f"[scene {scene_id}] loadScene done", flush=True)

        scene_dir = os.path.join(save_root, f"scene_{scene_id:03d}")
        os.makedirs(scene_dir, exist_ok=True)

        # Copy PLY for reference
        shutil.copy2(ply_src, os.path.join(scene_dir, "pointcloud.ply"))

        traj_count = 0
        attempts = 0
        max_attempts = args.trajs_per_scene * 4

        while traj_count < args.trajs_per_scene and attempts < max_attempts:
            attempts += 1

            pair = sample_pair(env, planner, MIN_DIST, MAX_DIST)
            if pair is None:
                continue

            sp, sv, sa, gp, gv, ga = pair
            print(f"[scene {scene_id}] plan attempt {attempts}: {sp} -> {gp}", flush=True)
            result = planner.plan(
                sp, sv, sa,
                gp, gv, ga,
            )
            print(f"[scene {scene_id}] plan done: success={result['success']}", flush=True)

            if not result["success"]:
                continue

            # Collect depth images along the trajectory
            waypoints = result["waypoints"].astype(np.float32)
            depths = collect_depths(env, flatmap, waypoints)

            # Save
            tdir = os.path.join(scene_dir, f"traj_{traj_count:04d}")
            os.makedirs(tdir, exist_ok=True)

            np.save(os.path.join(tdir, "coeffs.npy"),    result["coeffs"])
            np.save(os.path.join(tdir, "durations.npy"), result["durations"])
            np.save(os.path.join(tdir, "waypoints.npy"), waypoints)
            np.save(os.path.join(tdir, "depths.npy"),    depths)
            np.save(os.path.join(tdir, "start_pva.npy"), np.stack([sp, sv, sa]))
            np.save(os.path.join(tdir, "goal_pva.npy"),  np.stack([gp, gv, ga]))

            with open(os.path.join(tdir, "meta.json"), "w") as f:
                json.dump(
                    {
                        "scene_id": scene_id,
                        "traj_id": traj_count,
                        "total_duration": result["total_duration"],
                        "num_pieces": result["num_pieces"],
                        "dt": DT,
                    },
                    f,
                )

            traj_count += 1

        index["scenes"].append(
            {"scene_id": scene_id, "num_trajs": traj_count, "map_bound": map_bound}
        )
        index["total_trajectories"] += traj_count
        print(f"Scene {scene_id:3d}: {traj_count}/{args.trajs_per_scene} trajectories "
              f"({attempts} attempts)")

    with open(os.path.join(save_root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. {index['total_trajectories']} trajectories saved to {save_root}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect GCOPTER trajectory dataset")
    parser.add_argument("--num_scenes",      type=int, default=NUM_SCENES)
    parser.add_argument("--trajs_per_scene", type=int, default=TRAJS_PER_SCENE)
    parser.add_argument("--save_root",       type=str, default=SAVE_ROOT)
    main(parser.parse_args())
