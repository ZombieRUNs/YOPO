"""
Collect GCOPTER reference trajectory dataset using Flightmare scenes.

For each scene:
  1. Spawn random trees and export PLY point cloud.
  2. Load PLY into GCOPTER VoxelMap.
  3. Sample random start/goal pairs (collision-free via Flightmare reset).
  4. Plan MINCO trajectory with GCOPTER (C++ via Pybind11).
  5. Walk the trajectory at 30 Hz, set drone state, collect depth and RGB images.
  6. Save coeffs, durations, waypoints, depths.npy, rgb.mp4, and meta info.

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
        depths.npy       [M,H,W]     float16 (0..1 normalised, 20m clip)
        rgb.mp4          RGB video aligned with waypoint order (CFR)
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
import sys
import tempfile

import cv2
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
DEFAULT_IMG_WIDTH = 160
DEFAULT_IMG_HEIGHT = 96
DEFAULT_RGB_FPS = 30.0
DEFAULT_VMAX_MIN = 7.0
DEFAULT_VMAX_MAX = 7.0
DEFAULT_CAMERA_PITCH_DEG = 0.0

MAP_BOUND = [-25.0, 25.0, -25.0, 25.0, 0.0, 5.0]  # [xmin,xmax,ymin,ymax,zmin,zmax]
VOXEL_WIDTH = 0.25   # metres
DILATE_R = 0.5       # metres

# FLIGHTMARE_PATH defaults to the YOPO root (parent of this script's directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLIGHTMARE_PATH = os.environ.get("FLIGHTMARE_PATH",
                                  os.path.dirname(_SCRIPT_DIR))
PLY_DIR = os.path.join(FLIGHTMARE_PATH, "run", "yopo_sim")
CFG_PATH = os.path.join(FLIGHTMARE_PATH, "flightlib", "configs")
DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "collect_gcopter_trajs.yaml")


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


def quat_mul(q1, q2):
    """Quaternion multiply (wxyz): q = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


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


def save_rgb_video(rgbs: np.ndarray, out_path: str, fps: float) -> None:
    """Save RGB frames [M,H,W,3] uint8 as mp4."""
    if rgbs.ndim != 4 or rgbs.shape[-1] != 3:
        raise ValueError(f"rgbs must be [M,H,W,3], got {rgbs.shape}")
    m, h, w, _ = rgbs.shape
    if m == 0:
        raise ValueError("Cannot save empty RGB sequence.")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    for i in range(m):
        frame_rgb = rgbs[i]
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def collect_images(env, flatmap, waypoints, img_width, img_height, camera_pitch_deg=0.0):
    """
    Walk the trajectory and collect depth/RGB images at each waypoint.

    Args:
        env:        Flightmare FlightEnvVec
        flatmap:    gcopter_planner.FlatnessMap (pre-initialised)
        waypoints:  ndarray [M, 13]

    Returns:
        depths: ndarray [M, H, W] float16, normalised to [0, 1]
                (normalisation is done by vec_env_wrapper.getDepthImage)
        rgbs:   ndarray [M, H, W, 3] uint8
    """
    M = len(waypoints)
    depths = np.zeros((M, img_height, img_width), dtype=np.float16)
    rgbs = np.zeros((M, img_height, img_width, 3), dtype=np.uint8)

    pitch_rad = float(camera_pitch_deg) * math.pi / 180.0
    q_pitch = np.array([math.cos(0.5 * pitch_rad), 0.0, math.sin(0.5 * pitch_rad), 0.0], dtype=np.float64)

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

        if abs(pitch_rad) > 1e-12:
            quat = quat_mul(quat.astype(np.float64), q_pitch)
            n = np.linalg.norm(quat)
            if n > 1e-12:
                quat = quat / n

        # setState sets C++ state; render() syncs with Unity and fills image buffer
        env.setState(np.array([px, py, pz]), vel, acc, quat)
        env.render()

        # getDepthImage() returns [num_envs, 1, net_h, net_w], already normalised [0,1]
        depth = env.getDepthImage()[0][0]
        # getRGBImage(rgb=True) returns [num_envs, H*W*3], uint8.
        rgb_flat = env.getRGBImage(rgb=True)[0]
        rgb = np.reshape(rgb_flat, (env.img_height, env.img_width, 3))

        # Keep depth/rgb at a unified, configurable output resolution.
        if depth.shape != (img_height, img_width):
            depth = cv2.resize(depth.astype(np.float32), (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        if rgb.shape[:2] != (img_height, img_width):
            rgb = cv2.resize(rgb, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

        depths[k] = depth.astype(np.float16)
        rgbs[k] = rgb.astype(np.uint8)

    return depths, rgbs


def make_planner_with_vmax(gcopter_planner_mod, planner_cfg_path: str, v_max: float):
    """Create a fresh GcopterPlanner with overridden MaxVelMag."""
    cfg_obj = YAML(typ="safe").load(open(planner_cfg_path, "r"))
    if not isinstance(cfg_obj, dict):
        raise ValueError(f"Invalid planner config: {planner_cfg_path}")
    cfg_obj["MaxVelMag"] = float(v_max)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        tmp_cfg_path = tf.name
        YAML().dump(cfg_obj, tf)
    try:
        planner = gcopter_planner_mod.GcopterPlanner(tmp_cfg_path)
    finally:
        try:
            os.remove(tmp_cfg_path)
        except OSError:
            pass
    return planner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    import gcopter_planner
    from flightgym import QuadrotorEnv_v1
    from flightpolicy.envs import vec_env_wrapper as wrapper

    if args.img_width <= 0 or args.img_height <= 0:
        raise ValueError("--img_width and --img_height must be positive integers.")
    if args.min_dist <= 0 or args.max_dist <= args.min_dist:
        raise ValueError("Require 0 < min_dist < max_dist.")
    if args.rgb_fps <= 0:
        raise ValueError("--rgb_fps must be positive.")
    if args.v_max_min <= 0 or args.v_max_max < args.v_max_min:
        raise ValueError("Require 0 < v_max_min <= v_max_max.")
    if not np.isfinite(args.camera_pitch_deg):
        raise ValueError("--camera_pitch_deg must be finite.")

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
        "img_shape": [args.img_height, args.img_width],
        "rgb_shape": [args.img_height, args.img_width, 3],
        "rgb_storage": "rgb.mp4",
        "rgb_fps": args.rgb_fps,
        "img_range": "0-1 normalised, 20m clip",
        "scenes": [],
    }

    for scene_id in range(args.num_scenes):
        # Generate scene + export PLY
        print(f"[scene {scene_id}] spawnTrees", flush=True)
        env.setMapID(np.array([scene_id]))
        env.spawnTreesAndSavePointcloud(scene_id, spacing=3.16)
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

            pair = sample_pair(env, planner, args.min_dist, args.max_dist)
            if pair is None:
                continue

            sp, sv, sa, gp, gv, ga = pair
            v_max = float(np.random.uniform(args.v_max_min, args.v_max_max))
            planner_run = make_planner_with_vmax(gcopter_planner, planner_cfg, v_max)
            planner_run.loadScene(ply_src, map_bound, VOXEL_WIDTH, DILATE_R)
            print(
                f"[scene {scene_id}] plan attempt {attempts}: {sp} -> {gp} "
                f"(v_max={v_max:.3f})",
                flush=True,
            )
            result = planner_run.plan(
                sp, sv, sa,
                gp, gv, ga,
            )
            print(f"[scene {scene_id}] plan done: success={result['success']}", flush=True)

            if not result["success"]:
                continue

            # Collect depth/RGB images along the trajectory
            waypoints = result["waypoints"].astype(np.float32)
            depths, rgbs = collect_images(
                env, flatmap, waypoints,
                img_width=args.img_width,
                img_height=args.img_height,
                camera_pitch_deg=args.camera_pitch_deg,
            )

            # Save
            tdir = os.path.join(scene_dir, f"traj_{traj_count:04d}")
            os.makedirs(tdir, exist_ok=True)

            np.save(os.path.join(tdir, "coeffs.npy"),    result["coeffs"])
            np.save(os.path.join(tdir, "durations.npy"), result["durations"])
            np.save(os.path.join(tdir, "waypoints.npy"), waypoints)
            np.save(os.path.join(tdir, "depths.npy"),    depths)
            save_rgb_video(rgbs, os.path.join(tdir, "rgb.mp4"), args.rgb_fps)
            np.save(os.path.join(tdir, "start_pva.npy"), np.stack([sp, sv, sa]))
            np.save(os.path.join(tdir, "goal_pva.npy"),  np.stack([gp, gv, ga]))
            np.save(os.path.join(tdir, "v_max.npy"),     np.float32(v_max))

            with open(os.path.join(tdir, "meta.json"), "w") as f:
                json.dump(
                    {
                        "scene_id": scene_id,
                        "traj_id": traj_count,
                        "total_duration": result["total_duration"],
                        "num_pieces": result["num_pieces"],
                        "dt": DT,
                        "img_width": args.img_width,
                        "img_height": args.img_height,
                        "rgb_fps": args.rgb_fps,
                        "rgb_file": "rgb.mp4",
                        "v_max": v_max,
                        "camera_pitch_deg": args.camera_pitch_deg,
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


def load_yaml_config(config_path: str) -> dict:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = YAML(typ="safe").load(open(config_path, "r"))
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a dict: {config_path}")
    return cfg


def parse_args():
    argv = [a for a in sys.argv[1:] if not a.startswith("__")]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Collect GCOPTER trajectory dataset")
    parser.add_argument("--config",          type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--num_scenes",      type=int, default=NUM_SCENES)
    parser.add_argument("--trajs_per_scene", type=int, default=TRAJS_PER_SCENE)
    parser.add_argument("--save_root",       type=str, default=SAVE_ROOT)
    parser.add_argument("--img_width",       type=int, default=DEFAULT_IMG_WIDTH)
    parser.add_argument("--img_height",      type=int, default=DEFAULT_IMG_HEIGHT)
    parser.add_argument("--min_dist",        type=float, default=MIN_DIST)
    parser.add_argument("--max_dist",        type=float, default=MAX_DIST)
    parser.add_argument("--rgb_fps",         type=float, default=DEFAULT_RGB_FPS)
    parser.add_argument("--v_max_min",       type=float, default=DEFAULT_VMAX_MIN)
    parser.add_argument("--v_max_max",       type=float, default=DEFAULT_VMAX_MAX)
    parser.add_argument("--camera_pitch_deg", type=float, default=DEFAULT_CAMERA_PITCH_DEG)

    valid_keys = {a.dest for a in parser._actions}
    unknown = sorted([k for k in cfg.keys() if k not in valid_keys])
    if unknown:
        raise ValueError(f"Unknown keys in config {pre_args.config}: {unknown}")
    parser.set_defaults(**cfg)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
