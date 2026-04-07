"""
Collect SUPER planner trajectory dataset using Flightmare scenes.

For each scene (from existing gcopter_trajs dataset):
  1. Convert PLY → PCD (for SUPER's ROG-Map).
  2. Launch SUPER ROS stack (perfect_drone_sim + fsm_node) with this scene.
  3. Load same scene in Flightmare (for depth/RGB rendering).
  4. Collect N trajectories:
     a. Send goal via /goal, capture /planning_cmd/poly_traj.
     b. Extract waypoints from order-7 polynomial (30 Hz).
     c. Collect depth/RGB via Flightmare at each waypoint.
     d. Save data.
  5. Kill SUPER stack; next scene.

Drone teleportation: perfect_drone_sim uses cmdCallback which directly sets
position from /planning/pos_cmd — no physics. We exploit this to reset
drone position between trajectories.

Output layout:
  dataset/super_trajs/
    index.json
    scene_000/
      pointcloud.ply   (symlink to gcopter_trajs source)
      traj_0000/
        meta.json
        poly_coeffs.npy   [N, 3, 8] float64  (N pieces, XYZ, 8 coeffs desc: c7..c0)
        durations.npy     [N]        float64
        waypoints.npy     [M, 13]    float32  [t,px,py,pz,vx,vy,vz,ax,ay,az,jx,jy,jz]
        depths.npy        [M, H, W]  float16  (0-1 normalised, 20 m clip)
        rgb.mp4
        start_pva.npy     [3, 3]     float64
        goal_pos.npy      [3]        float64
"""

import argparse
import json
import math
import os
import atexit
import signal
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
from ruamel.yaml import YAML, RoundTripDumper, dump

# ---------------------------------------------------------------------------
# Path constants (inside Docker: /root/workspace == repo root)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_YOPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_WORKSPACE = os.path.dirname(_YOPO_ROOT)

SUPER_ROOT         = os.path.join(_WORKSPACE, "ROS", "src", "SUPER")
SUPER_FOREST_DIR   = os.path.join(SUPER_ROOT, "forest")
SUPER_PCD_LINK     = os.path.join(SUPER_ROOT, "mars_uav_sim", "perfect_drone_sim", "pcd", "forest.pcd")
SUPER_CLICK_CFG    = os.path.join(SUPER_ROOT, "mars_uav_sim", "perfect_drone_sim", "config", "click.yaml")
SUPER_PLANNER_CFG  = os.path.join(SUPER_ROOT, "super_planner", "config", "click_smooth_ros1.yaml")
FLIGHTMARE_BIN     = os.path.join(_YOPO_ROOT, "flightrender", "RPG_Flightmare", "flightmare.x86_64")
FLIGHTLIB_CFG      = os.path.join(_YOPO_ROOT, "flightlib", "configs")
YOPO_SIM_DIR       = os.path.join(_SCRIPT_DIR, "yopo_sim")

GCOPTER_DATASET_ROOT = "dataset/gcopter_trajs"
SUPER_DATASET_ROOT   = "dataset/super_trajs"

DT       = 1.0 / 30.0
MIN_DIST = 30.0
MAX_DIST = 40.0

DEFAULT_INIT_POS = [-10.0, 20.0, 2.5]   # fallback when env.reset() finds nothing


def clamp_flight_z(pos: np.ndarray, safe_z: float) -> np.ndarray:
    """Return a copy with z clamped to a safe fixed flight height."""
    out = np.array(pos, dtype=np.float64).copy()
    out[2] = float(safe_z)
    return out


def within_scene_bounds(pos: np.ndarray, center, half, margin: float = 5.0) -> bool:
    """Return True if pos XY is at least `margin` metres inside the scene bounding box.

    center / half come from quadrotor_env.yaml bounding_box_origin / (0.5 * bounding_box).
    Only XY is checked; Z is handled separately by clamp_flight_z.
    """
    for i in range(2):
        lo = center[i] - half[i] + margin
        hi = center[i] + half[i] - margin
        if not (lo <= pos[i] <= hi):
            return False
    return True


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _yaml_load_rw(path):
    return YAML().load(open(path))


def _yaml_dump_rw(data, path):
    with open(path, "w") as f:
        YAML().dump(data, f)


def load_super_runtime_params():
    """Load key runtime params from SUPER configs for interface alignment."""
    planner_cfg = _yaml_load_rw(SUPER_PLANNER_CFG)
    click_cfg = _yaml_load_rw(SUPER_CLICK_CFG)
    robot_r = float(planner_cfg["super_planner"]["robot_r"])
    click_h = float(click_cfg.get("fsm", {}).get("click_height", 1.5))
    return {"robot_r": robot_r, "click_height": click_h}


def update_super_scene(scene_id: int, init_pos, pcd_dir: str, super_robot_r=None, super_max_vel=None,
                       super_planning_horizon=None, super_receding_dis=None,
                       super_corridor_line_max_length=None,
                       super_safe_corridor_line_max_length=None, super_corridor_bound_dis=None) -> None:
    """Update symlink + both YAML configs for a new scene."""
    pcd_src = os.path.join(pcd_dir, f"scene_{scene_id:03d}.pcd")

    # Symlink
    if os.path.islink(SUPER_PCD_LINK):
        os.remove(SUPER_PCD_LINK)
    elif os.path.exists(SUPER_PCD_LINK):
        os.remove(SUPER_PCD_LINK)
    os.symlink(pcd_src, SUPER_PCD_LINK)

    # click.yaml — init_position
    cfg = _yaml_load_rw(SUPER_CLICK_CFG)
    cfg["init_position"]["x"] = float(init_pos[0])
    cfg["init_position"]["y"] = float(init_pos[1])
    cfg["init_position"]["z"] = float(init_pos[2])
    _yaml_dump_rw(cfg, SUPER_CLICK_CFG)

    # click_smooth_ros1.yaml — rog_map.pcd_name + planner overrides
    cfg2 = _yaml_load_rw(SUPER_PLANNER_CFG)
    cfg2["rog_map"]["pcd_name"] = pcd_src
    if super_robot_r is not None:
        cfg2["super_planner"]["robot_r"] = float(super_robot_r)
    if super_max_vel is not None:
        cfg2["traj_opt"]["boundary"]["max_vel"] = float(super_max_vel)
    if super_planning_horizon is not None:
        cfg2["super_planner"]["planning_horizon"] = float(super_planning_horizon)
    if super_receding_dis is not None:
        cfg2["super_planner"]["receding_dis"] = float(super_receding_dis)
    if super_corridor_line_max_length is not None:
        cfg2["super_planner"]["corridor_line_max_length"] = float(super_corridor_line_max_length)
    if super_safe_corridor_line_max_length is not None:
        cfg2["super_planner"]["safe_corridor_line_max_length"] = float(super_safe_corridor_line_max_length)
    if super_corridor_bound_dis is not None:
        cfg2["super_planner"]["corridor_bound_dis"] = float(super_corridor_bound_dis)
    _yaml_dump_rw(cfg2, SUPER_PLANNER_CFG)


def convert_ply_to_pcd(ply_path: str, pcd_path: str) -> bool:
    """Convert PLY to PCD via pcl_ply2pcd. Returns True on success."""
    if os.path.exists(pcd_path):
        return True
    os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
    r = subprocess.run(["pcl_ply2pcd", ply_path, pcd_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] pcl_ply2pcd failed for {ply_path}: {r.stderr.strip()}")
        return False
    return True


# ---------------------------------------------------------------------------
# Polynomial utilities
# ---------------------------------------------------------------------------

def _eval_piece(coeffs_xyz, t):
    """Evaluate pos/vel/acc/jerk of order-7 piece at local time t.

    Args:
        coeffs_xyz: [3, 8] float64, descending order (c7*t^7 + ... + c0)
        t: float

    Returns:
        pos, vel, acc, jer: each [3] float64
    """
    pos = np.array([np.polyval(coeffs_xyz[d],                     t) for d in range(3)])
    vel = np.array([np.polyval(np.polyder(coeffs_xyz[d]),          t) for d in range(3)])
    acc = np.array([np.polyval(np.polyder(coeffs_xyz[d], 2),       t) for d in range(3)])
    jer = np.array([np.polyval(np.polyder(coeffs_xyz[d], 3),       t) for d in range(3)])
    return pos, vel, acc, jer


def poly_traj_to_waypoints(poly_coeffs, durations, dt=DT):
    """Sample order-7 piecewise polynomial at fixed dt.

    Args:
        poly_coeffs: [N, 3, 8] float64
        durations:   [N]       float64

    Returns:
        waypoints:   [M, 13]   float32  [t, px,py,pz, vx,vy,vz, ax,ay,az, jx,jy,jz]
    """
    rows = []
    t_abs = 0.0
    for i, T in enumerate(durations):
        coeffs = poly_coeffs[i]          # [3, 8]
        ts = np.arange(0.0, T - 1e-8, dt)
        for t in ts:
            pos, vel, acc, jer = _eval_piece(coeffs, t)
            rows.append(np.concatenate([[t_abs + t], pos, vel, acc, jer]))
        t_abs += T
    if not rows:
        return np.zeros((0, 13), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def cmd_history_to_waypoints(cmd_hist: np.ndarray, dt=DT) -> np.ndarray:
    """Convert /planning/pos_cmd history to fixed-rate waypoints [M,13]."""
    if cmd_hist.shape[0] < 2:
        return np.zeros((0, 13), dtype=np.float32)

    # Keep monotonic timestamps and drop duplicates.
    ts = cmd_hist[:, 0]
    keep = np.ones(len(ts), dtype=bool)
    keep[1:] = ts[1:] > ts[:-1]
    hist = cmd_hist[keep]
    if hist.shape[0] < 2:
        return np.zeros((0, 13), dtype=np.float32)

    t = hist[:, 0] - hist[0, 0]
    t_end = float(t[-1])
    if t_end <= 1e-6:
        return np.zeros((0, 13), dtype=np.float32)

    t_grid = np.arange(0.0, t_end + 1e-9, dt, dtype=np.float64)
    if t_grid.size < 2:
        return np.zeros((0, 13), dtype=np.float32)

    pos = np.zeros((t_grid.size, 3), dtype=np.float64)
    vel = np.zeros((t_grid.size, 3), dtype=np.float64)
    acc = np.zeros((t_grid.size, 3), dtype=np.float64)
    jer = np.zeros((t_grid.size, 3), dtype=np.float64)
    for d in range(3):
        pos[:, d] = np.interp(t_grid, t, hist[:, 1 + d])
        vel[:, d] = np.interp(t_grid, t, hist[:, 4 + d])
        acc[:, d] = np.interp(t_grid, t, hist[:, 7 + d])
        jer[:, d] = np.interp(t_grid, t, hist[:, 10 + d])

    return np.concatenate(
        [t_grid[:, None], pos, vel, acc, jer], axis=1
    ).astype(np.float32)


def decode_poly_traj_msg(msg):
    """Decode quadrotor_msgs/PolynomialTrajectory.

    Returns:
        poly_coeffs: [N, 3, 8] float64
        durations:   [N]       float64
        total_dur:   float
    """
    N       = msg.piece_num_pos
    n_coeff = msg.order_pos + 1   # should be 8

    coefs_x = np.array(msg.coef_pos_x[: N * n_coeff], dtype=np.float64).reshape(N, n_coeff)
    coefs_y = np.array(msg.coef_pos_y[: N * n_coeff], dtype=np.float64).reshape(N, n_coeff)
    coefs_z = np.array(msg.coef_pos_z[: N * n_coeff], dtype=np.float64).reshape(N, n_coeff)
    durations = np.array(msg.time_pos[:N], dtype=np.float64)

    poly_coeffs = np.stack([coefs_x, coefs_y, coefs_z], axis=1)   # [N, 3, 8]
    return poly_coeffs, durations, float(durations.sum())


# ---------------------------------------------------------------------------
# Flightmare launch helpers (mirrored from test_flowpilot_px4.py)
# ---------------------------------------------------------------------------

def _build_flightmare_env():
    env = os.environ.copy()
    if env.get('DISPLAY'):
        env.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
        env.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
        env.setdefault('__VK_LAYER_NV_optimus', 'NVIDIA_only')
    return env


def _cleanup_stale_flightmare(binary_name: str = 'flightmare.x86_64'):
    result = subprocess.run(['pgrep', '-f', binary_name],
                            capture_output=True, text=True, check=False)
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    if not pids:
        return
    print(f'[INFO] terminating stale Flightmare: {pids}', flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 5.0
    while time.time() < deadline:
        alive = [p for p in pids if _pid_alive(p)]
        if not alive:
            return
        time.sleep(0.2)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


_STUB_DLOPEN_SRC = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
void *dlopen(const char *f, int flags) {
    static void *(*real)(const char*,int)=NULL;
    if(!real)real=dlsym(RTLD_NEXT,"dlopen");
    if(f&&strstr(f,"libasound"))return NULL;
    return real(f,flags);
}
"""


def _build_alsa_stub() -> str:
    so_path  = '/tmp/stub_dlopen_flightmare.so'
    src_path = '/tmp/stub_dlopen_flightmare.c'
    if not os.path.exists(so_path):
        with open(src_path, 'w') as f:
            f.write(_STUB_DLOPEN_SRC)
        ret = subprocess.run(
            ['gcc', '-shared', '-fPIC', '-o', so_path, src_path, '-ldl'],
            capture_output=True, text=True)
        if ret.returncode != 0:
            print(f'[WARN] ALSA stub compile failed: {ret.stderr}', flush=True)
            return ''
        print('[INFO] compiled ALSA dlopen stub.', flush=True)
    return so_path


def launch_flightmare(binary_path: str):
    """Launch Flightmare with NVIDIA offload hints and ALSA dlopen stub."""
    _cleanup_stale_flightmare(os.path.basename(binary_path))
    env  = _build_flightmare_env()
    stub = _build_alsa_stub()
    if stub:
        preload = env.get('LD_PRELOAD', '')
        env['LD_PRELOAD'] = f'{stub}:{preload}' if preload else stub
    proc = subprocess.Popen(
        [binary_path],
        cwd=os.path.dirname(binary_path),
        env=env,
        start_new_session=True)

    def _cleanup():
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

    atexit.register(_cleanup)
    return proc


# ---------------------------------------------------------------------------
# Image helpers (mirrored from collect_gcopter_trajs.py)
# ---------------------------------------------------------------------------

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def collect_images(env, flatmap, waypoints, img_width, img_height,
                   camera_pitch_deg=0.0, yaw_bias=0.0):
    """Walk trajectory and collect depth/RGB via Flightmare.

    Args:
        yaw_bias: constant yaw offset (rad) added to flatness yaw for the
                  entire trajectory.  Sampled once per trajectory when
                  --yaw_rand_deg > 0, simulating the gap between the
                  flatness-derived yaw and the real PX4 yaw controller.
    """
    M = len(waypoints)
    depths = np.zeros((M, img_height, img_width), dtype=np.float16)
    rgbs   = np.zeros((M, img_height, img_width, 3), dtype=np.uint8)

    pitch_rad = float(camera_pitch_deg) * math.pi / 180.0
    q_pitch   = np.array([math.cos(0.5 * pitch_rad), 0.0,
                           math.sin(0.5 * pitch_rad), 0.0], dtype=np.float64)

    for k in range(M):
        t, px, py, pz, vx, vy, vz, ax, ay, az, jx, jy, jz = waypoints[k]
        vel = np.array([vx, vy, vz]);  acc = np.array([ax, ay, az])
        jer = np.array([jx, jy, jz])

        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed < 1e-3:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            psi = math.atan2(vy, vx) + yaw_bias
            _thr, quat, _omg = flatmap.forward(vel, acc, jer, psi, 0.0)
            quat = np.array(quat)

        if abs(pitch_rad) > 1e-12:
            quat = _quat_mul(quat.astype(np.float64), q_pitch)
            n = np.linalg.norm(quat)
            if n > 1e-12:
                quat /= n

        env.setState(np.array([px, py, pz]), vel, acc, quat)
        env.render()

        depth    = env.getDepthImage()[0][0]
        rgb_flat = env.getRGBImage(rgb=True)[0]
        rgb      = np.reshape(rgb_flat, (env.img_height, env.img_width, 3))

        if depth.shape != (img_height, img_width):
            depth = cv2.resize(depth.astype(np.float32),
                               (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        if rgb.shape[:2] != (img_height, img_width):
            rgb = cv2.resize(rgb, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

        depths[k] = depth.astype(np.float16)
        rgbs[k]   = rgb.astype(np.uint8)

    return depths, rgbs


def save_rgb_video(rgbs: np.ndarray, out_path: str, fps: float) -> None:
    m, h, w, _ = rgbs.shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    for i in range(m):
        writer.write(cv2.cvtColor(rgbs[i], cv2.COLOR_RGB2BGR))
    writer.release()


# ---------------------------------------------------------------------------
# ROS interaction
# ---------------------------------------------------------------------------

class SuperCollector:
    """Handles goal publishing, trajectory capture, and drone teleportation."""

    def __init__(self):
        import rospy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from quadrotor_msgs.msg import PolynomialTrajectory, PositionCommand

        self._lock_traj = threading.Lock()
        self._lock_odom = threading.Lock()
        self._lock_cmd = threading.Lock()
        self._latest_traj = None
        self._latest_odom = None
        self._cmd_history = []

        self._goal_pub    = rospy.Publisher("/goal",              PoseStamped,          queue_size=1)
        self._cmd_pub     = rospy.Publisher("/planning/pos_cmd",  PositionCommand,      queue_size=1)
        self._cmd_sub     = rospy.Subscriber("/planning/pos_cmd",
                                             PositionCommand, self._cmd_cb)
        self._traj_sub    = rospy.Subscriber("/planning_cmd/poly_traj",
                                             PolynomialTrajectory, self._traj_cb)
        self._odom_sub    = rospy.Subscriber("/lidar_slam/odom",
                                             Odometry, self._odom_cb)

    def reset_state(self):
        """Clear cached messages between scenes (subscribers persist)."""
        with self._lock_traj:
            self._latest_traj = None
        with self._lock_odom:
            self._latest_odom = None
        with self._lock_cmd:
            self._cmd_history = []

    # ---- callbacks ----

    def _traj_cb(self, msg):
        # piece_num_pos == 0 → heartbeat only, no actual trajectory
        if msg.piece_num_pos == 0:
            return
        with self._lock_traj:
            self._latest_traj = msg

    def _odom_cb(self, msg):
        with self._lock_odom:
            self._latest_odom = msg

    def _cmd_cb(self, msg):
        ts = float(msg.header.stamp.to_sec())
        if ts <= 0.0:
            ts = time.time()
        row = np.array([
            ts,
            float(msg.position.x), float(msg.position.y), float(msg.position.z),
            float(msg.velocity.x), float(msg.velocity.y), float(msg.velocity.z),
            float(msg.acceleration.x), float(msg.acceleration.y), float(msg.acceleration.z),
            float(msg.jerk.x), float(msg.jerk.y), float(msg.jerk.z),
        ], dtype=np.float64)
        with self._lock_cmd:
            self._cmd_history.append(row)

    # ---- queries ----

    def get_position(self):
        """Returns current drone position [3] or None."""
        with self._lock_odom:
            if self._latest_odom is None:
                return None
            p = self._latest_odom.pose.pose.position
            return np.array([p.x, p.y, p.z])

    def wait_for_odom(self, timeout=60.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.get_position() is not None:
                return True
            time.sleep(0.2)
        return False

    def clear_cmd_history(self) -> None:
        with self._lock_cmd:
            self._cmd_history = []

    def get_cmd_history(self):
        with self._lock_cmd:
            if not self._cmd_history:
                return np.zeros((0, 13), dtype=np.float64)
            return np.array(self._cmd_history, dtype=np.float64)

    # ---- actions ----

    def teleport(self, pos) -> None:
        """Move drone instantly by injecting /planning/pos_cmd (zero vel/acc)."""
        import rospy
        from quadrotor_msgs.msg import PositionCommand
        cmd = PositionCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.position.x, cmd.position.y, cmd.position.z = float(pos[0]), float(pos[1]), float(pos[2])
        cmd.velocity.x = cmd.velocity.y = cmd.velocity.z = 0.0
        cmd.acceleration.x = cmd.acceleration.y = cmd.acceleration.z = 0.0
        cmd.yaw = 0.0
        self._cmd_pub.publish(cmd)
        time.sleep(0.3)   # let odometry update

    def send_goal(self, goal_pos) -> None:
        import rospy
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(goal_pos[0])
        msg.pose.position.y = float(goal_pos[1])
        msg.pose.position.z = float(goal_pos[2])
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)

    def wait_for_new_traj(self, timeout=15.0, min_duration=0.5):
        """Wait for the next real /planning_cmd/poly_traj (piece_num_pos > 0).

        Returns decoded (poly_coeffs, durations, total_dur) or None on timeout.
        """
        with self._lock_traj:
            self._latest_traj = None

        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock_traj:
                msg = self._latest_traj
                if msg is not None:
                    if msg.piece_num_pos > 0:
                        result = decode_poly_traj_msg(msg)
                        if result[2] >= min_duration:
                            return result
                    # discard heartbeat / degenerate message, keep waiting
                    self._latest_traj = None
            time.sleep(0.05)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    import rospy
    import gcopter_planner
    from flightgym import QuadrotorEnv_v1
    from flightpolicy.envs import vec_env_wrapper as wrapper

    save_root    = args.save_root
    gcopter_root = args.gcopter_trajs_root
    os.makedirs(save_root,       exist_ok=True)
    os.makedirs(SUPER_FOREST_DIR, exist_ok=True)

    # Backward-compatible alias override (matches collect_gcopter_trajs semantics)
    if args.min_start_goal_dist is not None:
        args.min_dist = float(args.min_start_goal_dist)
    if args.max_start_goal_dist is not None:
        args.max_dist = float(args.max_start_goal_dist)

    if args.min_dist <= 0 or args.max_dist <= args.min_dist:
        raise ValueError("Require 0 < min_dist < max_dist.")
    if args.goal_reach_dist <= 0:
        raise ValueError("--goal_reach_dist must be positive.")
    if args.goal_timeout <= 0:
        raise ValueError("--goal_timeout must be positive.")
    if args.max_replan_segments <= 0:
        raise ValueError("--max_replan_segments must be positive.")

    # Read SUPER runtime params from configs and align collector checks.
    super_params = load_super_runtime_params()
    super_robot_r = float(super_params["robot_r"])
    if args.super_robot_r is not None:
        super_robot_r = float(args.super_robot_r)
    if args.safe_flight_z is None:
        args.safe_flight_z = float(super_params["click_height"])
    if args.collision_check_r is None:
        args.collision_check_r = super_robot_r

    # ------------------------------------------------------------------ #
    #  Flightmare + GCOPTER (initialised once for the full run)           #
    # ------------------------------------------------------------------ #
    # Read scene bounding box for boundary-margin filtering of env.reset() samples.
    quad_env_cfg = YAML().load(open(os.path.join(FLIGHTLIB_CFG, "quadrotor_env.yaml")))
    _bb      = quad_env_cfg["quadrotor_env"]["bounding_box"]
    _bb_orig = quad_env_cfg["quadrotor_env"]["bounding_box_origin"]
    scene_center = np.array([float(_bb_orig[i]) for i in range(2)])
    scene_half   = np.array([0.5 * float(_bb[i])  for i in range(2)])
    xy_margin    = 5.0   # metres kept away from scene edge

    vec_cfg_path = os.path.join(FLIGHTLIB_CFG, "vec_env.yaml")
    cfg = YAML().load(open(vec_cfg_path))
    cfg["env"]["num_envs"]    = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"]      = True
    cfg["env"]["supervised"]  = False
    cfg["env"]["imitation"]   = False
    launch_flightmare(FLIGHTMARE_BIN)
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env.connectUnity()

    planner_cfg = os.path.join(FLIGHTLIB_CFG, "gcopter_config.yaml")
    planner     = gcopter_planner.GcopterPlanner(planner_cfg)

    flat_cfg = planner.getConfig()
    pp       = flat_cfg["physical_params"]
    flatmap  = gcopter_planner.FlatnessMap()
    flatmap.reset(pp[0], pp[1], pp[2], pp[3], pp[4], pp[5])

    # ------------------------------------------------------------------ #
    #  roscore + rospy node (initialised once)                            #
    # ------------------------------------------------------------------ #
    roscore_proc = subprocess.Popen(["roscore"],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
    time.sleep(2.0)
    rospy.init_node("super_collector", anonymous=True, disable_signals=True)
    collector = SuperCollector()

    # ------------------------------------------------------------------ #
    #  Dataset index                                                      #
    # ------------------------------------------------------------------ #
    index = {
        "total_scenes":       args.num_scenes,
        "total_trajectories": 0,
        "dt":         DT,
        "poly_order": 7,
        "planner":    "super",
        "img_shape":  [args.img_height, args.img_width],
        "rgb_fps":    args.rgb_fps,
        "scenes":     [],
    }

    ros_proc = None

    try:
        for scene_id in range(args.num_scenes):
            # ---- Rebuild Flightmare scene and use its generated PLY ----
            print(f"[scene {scene_id:03d}] Loading Flightmare scene ...", flush=True)
            env.setMapID(np.array([scene_id]))
            env.spawnTreesAndSavePointcloud(scene_id, spacing=4.0)
            ply_path = os.path.join(YOPO_SIM_DIR, f"pointcloud-{scene_id}.ply")
            if not os.path.exists(ply_path):
                print(f"[SKIP] scene {scene_id}: runtime pointcloud not found: {ply_path}", flush=True)
                continue
            print(f"  using runtime pointcloud: {ply_path}", flush=True)

            # ---- PLY → PCD (from same map used by Flightmare rendering) ----
            pcd_path = os.path.join(SUPER_FOREST_DIR, f"scene_{scene_id:03d}.pcd")
            print(f"[scene {scene_id:03d}] Converting PLY → PCD ...", flush=True)
            if os.path.exists(pcd_path):
                os.remove(pcd_path)
            if not convert_ply_to_pcd(ply_path, pcd_path):
                print(f"[SKIP] scene {scene_id}: conversion failed", flush=True)
                continue

            # ---- Find init position for SUPER via env.reset() ----
            # env.reset() guarantees collision-free positions (C++ do-while + KdTree).
            # Additionally filter to keep positions ≥ xy_margin from scene edge.
            for _ in range(200):
                obs = env.reset()
                _p  = obs[0, 0:3]
                if within_scene_bounds(_p, scene_center, scene_half, xy_margin):
                    break
            init_pos = clamp_flight_z(obs[0, 0:3].copy(), args.safe_flight_z)

            # ---- Update SUPER configs ----
            update_super_scene(scene_id, init_pos, SUPER_FOREST_DIR,
                               super_robot_r=super_robot_r, super_max_vel=args.super_max_vel,
                               super_planning_horizon=args.super_planning_horizon,
                               super_receding_dis=args.super_receding_dis,
                               super_corridor_line_max_length=args.super_corridor_line_max_length,
                               super_safe_corridor_line_max_length=args.super_safe_corridor_line_max_length,
                               super_corridor_bound_dis=args.super_corridor_bound_dis)

            # ---- Launch SUPER ----
            if ros_proc is not None:
                ros_proc.terminate()
                ros_proc.wait()
                time.sleep(1.0)

            print(f"[scene {scene_id:03d}] Launching SUPER stack ...", flush=True)
            ros_proc = subprocess.Popen(
                ["roslaunch", "mission_planner", "click_demo.launch"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            collector.reset_state()

            print(f"[scene {scene_id:03d}] Waiting for odometry ...", flush=True)
            if not collector.wait_for_odom(timeout=60.0):
                print(f"  [ERROR] no odometry after 60s, skipping scene", flush=True)
                ros_proc.terminate(); ros_proc.wait(); ros_proc = None
                continue

            # Extra wait for ROG-Map to finish loading PCD (~3M pts)
            time.sleep(args.startup_wait)
            print(f"[scene {scene_id:03d}] SUPER ready.", flush=True)

            # ---- Prepare scene output dir ----
            scene_out_dir = os.path.join(save_root, f"scene_{scene_id:03d}")
            os.makedirs(scene_out_dir, exist_ok=True)
            ply_link = os.path.join(scene_out_dir, "pointcloud.ply")
            if not os.path.exists(ply_link):
                os.symlink(os.path.abspath(ply_path), ply_link)

            # ---- Trajectory collection loop ----
            traj_count        = 0
            attempts          = 0
            consec_failures   = 0
            max_att           = args.trajs_per_scene * 5
            current_pos = collector.get_position()
            if current_pos is not None:
                current_pos = clamp_flight_z(current_pos, args.safe_flight_z)

            while traj_count < args.trajs_per_scene and attempts < max_att:
                attempts += 1

                # Teleport if too many consecutive failures (drone likely stuck)
                # or every reset_interval successful trajectories.
                need_teleport = (consec_failures >= 2) or \
                                (traj_count > 0 and traj_count % args.reset_interval == 0)
                if need_teleport:
                    for _ in range(200):
                        _ns = env.reset()[0, 0:3]
                        if within_scene_bounds(_ns, scene_center, scene_half, xy_margin):
                            break
                    new_start = clamp_flight_z(_ns, args.safe_flight_z)
                    collector.teleport(new_start)
                    time.sleep(1.0)
                    cp = collector.get_position()
                    current_pos = clamp_flight_z(cp, args.safe_flight_z) if cp is not None else new_start
                    consec_failures = 0

                # Sample goal within distance range; env.reset() is already collision-free.
                # Also require goal to be ≥ xy_margin from scene edge (avoids air-wall hits).
                start_pos = clamp_flight_z(
                    current_pos.copy() if current_pos is not None else init_pos.copy(),
                    args.safe_flight_z)
                goal_pos = None
                for _ in range(300):
                    _gp = env.reset()[0, 0:3]
                    if not within_scene_bounds(_gp, scene_center, scene_half, xy_margin):
                        continue
                    gp = clamp_flight_z(_gp, args.safe_flight_z)
                    if args.min_dist <= np.linalg.norm(gp - start_pos) <= args.max_dist:
                        goal_pos = gp
                        break

                if goal_pos is None:
                    print(f"  [attempt {attempts}] failed to sample valid goal", flush=True)
                    continue

                goal_dist = float(np.linalg.norm(goal_pos - start_pos))
                start_pva = np.stack([start_pos, np.zeros(3), np.zeros(3)])

                print(f"  [traj {traj_count:04d}] {start_pos.round(1)} → {goal_pos.round(1)}"
                      f" (dist={goal_dist:.1f}m)",
                      flush=True)
                collector.clear_cmd_history()
                collector.send_goal(goal_pos)

                # Execution-trace mode: treat SUPER replans as internal details.
                # We only care about executed command stream and final goal reach.
                reached_goal = False
                dist_to_goal = float("inf")
                deadline = time.time() + args.goal_timeout
                while time.time() < deadline:
                    time.sleep(0.05)
                    current_pos = collector.get_position()
                    if current_pos is None:
                        continue
                    dist_to_goal = float(np.linalg.norm(goal_pos - current_pos))
                    if dist_to_goal <= args.goal_reach_dist:
                        reached_goal = True
                        break

                if not reached_goal:
                    consec_failures += 1
                    print(f"  [attempt {attempts}] did not reach goal (remain={dist_to_goal:.2f}m), skip"
                          f"{' [teleport next]' if consec_failures >= 2 else ''}", flush=True)
                    continue

                cmd_hist = collector.get_cmd_history()
                waypoints = cmd_history_to_waypoints(cmd_hist, dt=DT)
                if len(waypoints) < 2:
                    print(f"  [attempt {attempts}] insufficient command samples, skip", flush=True)
                    continue
                total_dur = float(waypoints[-1, 0] - waypoints[0, 0]) if len(waypoints) > 1 else 0.0
                poly_coeffs = np.zeros((0, 3, 8), dtype=np.float64)
                durations = np.zeros((0,), dtype=np.float64)

                # Collect depth/RGB via Flightmare
                yaw_bias = (np.random.uniform(-1, 1) * math.radians(args.yaw_rand_deg)
                            if args.yaw_rand_deg > 0 else 0.0)
                depths, rgbs = collect_images(
                    env, flatmap, waypoints,
                    img_width=args.img_width,
                    img_height=args.img_height,
                    camera_pitch_deg=args.camera_pitch_deg,
                    yaw_bias=yaw_bias,
                )

                # Save
                tdir = os.path.join(scene_out_dir, f"traj_{traj_count:04d}")
                os.makedirs(tdir, exist_ok=True)

                np.save(os.path.join(tdir, "poly_coeffs.npy"), poly_coeffs)
                np.save(os.path.join(tdir, "durations.npy"),   durations)
                np.save(os.path.join(tdir, "waypoints.npy"),   waypoints)
                np.save(os.path.join(tdir, "depths.npy"),      depths)
                save_rgb_video(rgbs, os.path.join(tdir, "rgb.mp4"), args.rgb_fps)
                np.save(os.path.join(tdir, "start_pva.npy"),   start_pva)
                np.save(os.path.join(tdir, "goal_pos.npy"),    goal_pos)
                np.save(os.path.join(tdir, "yaw_bias.npy"),   np.float32(yaw_bias))

                with open(os.path.join(tdir, "meta.json"), "w") as f:
                    json.dump({
                        "scene_id":       scene_id,
                        "traj_id":        traj_count,
                        "planner":        "super",
                        "poly_order":     7,
                        "total_duration": total_dur,
                        "num_pieces":     int(len(durations)),
                        "num_replan_segments": 0,
                        "goal_reach_dist": args.goal_reach_dist,
                        "goal_remain_dist": dist_to_goal,
                        "super_robot_r": super_robot_r,
                        "super_max_vel": args.super_max_vel,
                        "super_planning_horizon": args.super_planning_horizon,
                        "super_receding_dis": args.super_receding_dis,
                        "super_corridor_line_max_length": args.super_corridor_line_max_length,
                        "super_safe_corridor_line_max_length": args.super_safe_corridor_line_max_length,
                        "super_corridor_bound_dis": args.super_corridor_bound_dis,
                        "collision_check_r": args.collision_check_r,
                        "waypoint_source": "planning_pos_cmd",
                        "dt":             DT,
                        "img_width":      args.img_width,
                        "img_height":     args.img_height,
                        "rgb_fps":        args.rgb_fps,
                        "camera_pitch_deg": args.camera_pitch_deg,
                    }, f, indent=2)

                traj_count += 1
                consec_failures = 0
                print(f"  [traj {traj_count-1:04d}] saved  "
                      f"(exec-trace, {len(durations)} pieces, {total_dur:.1f}s, "
                      f"{len(waypoints)} waypoints)", flush=True)
                current_pos = collector.get_position()
                if current_pos is not None:
                    current_pos = clamp_flight_z(current_pos, args.safe_flight_z)

            # ---- Scene done ----
            index["scenes"].append({
                "scene_id":  scene_id,
                "num_trajs": traj_count,
                "attempts":  attempts,
            })
            index["total_trajectories"] += traj_count
            print(f"[scene {scene_id:03d}] {traj_count}/{args.trajs_per_scene} trajs "
                  f"({attempts} attempts)", flush=True)

    finally:
        # ---- Cleanup ----
        if ros_proc is not None:
            ros_proc.terminate()
            ros_proc.wait()
        roscore_proc.terminate()

    with open(os.path.join(save_root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. {index['total_trajectories']} trajectories → {save_root}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "collect_super_trajs.yaml")


def load_yaml_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    cfg = YAML(typ="safe").load(open(path))
    return cfg if isinstance(cfg, dict) else {}


def parse_args():
    argv = [a for a in sys.argv[1:] if not a.startswith("__")]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Collect SUPER trajectory dataset")
    parser.add_argument("--config",             type=str,   default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--num_scenes",         type=int,   default=36)
    parser.add_argument("--trajs_per_scene",    type=int,   default=20)
    parser.add_argument("--gcopter_trajs_root", type=str,   default=GCOPTER_DATASET_ROOT)
    parser.add_argument("--save_root",          type=str,   default=SUPER_DATASET_ROOT)
    parser.add_argument("--img_width",          type=int,   default=160)
    parser.add_argument("--img_height",         type=int,   default=96)
    parser.add_argument("--min_dist",           type=float, default=MIN_DIST)
    parser.add_argument("--max_dist",           type=float, default=MAX_DIST)
    parser.add_argument("--min_start_goal_dist", type=float, default=None,
                        help="Alias of --min_dist (for GCOPTER-style naming)")
    parser.add_argument("--max_start_goal_dist", type=float, default=None,
                        help="Alias of --max_dist (for GCOPTER-style naming)")
    parser.add_argument("--rgb_fps",            type=float, default=30.0)
    parser.add_argument("--camera_pitch_deg",   type=float, default=0.0)
    parser.add_argument("--yaw_rand_deg",       type=float, default=0.0,
                        help="Per-trajectory yaw randomisation range (deg). "
                             "Each trajectory gets one uniform sample in "
                             "[-yaw_rand_deg, +yaw_rand_deg] added to the "
                             "flatness yaw. 0 = disabled (default).")
    # SUPER-specific
    parser.add_argument("--startup_wait",  type=float, default=8.0,
                        help="Extra seconds after odom appears for ROG-Map PCD loading")
    parser.add_argument("--plan_timeout",  type=float, default=15.0,
                        help="Seconds to wait for /planning_cmd/poly_traj after sending goal")
    parser.add_argument("--exec_buffer",   type=float, default=2.0,
                        help="Extra seconds after each segment duration before checking goal distance")
    parser.add_argument("--reset_interval", type=int,  default=5,
                        help="Teleport drone to a fresh random start every N trajectories")
    parser.add_argument("--safe_flight_z", type=float, default=None,
                        help="Clamp SUPER init/start/goal z to this safe flight height (meters); "
                             "default: SUPER click_height")
    parser.add_argument("--collision_check_r", type=float, default=None,
                        help="Occupancy check radius for sampling/save filtering (meters); "
                             "default: SUPER super_planner.robot_r")
    parser.add_argument("--super_robot_r", type=float, default=None,
                        help="Override SUPER super_planner.robot_r before launching each scene")
    parser.add_argument("--super_max_vel", type=float, default=None,
                        help="Override SUPER traj_opt.boundary.max_vel (m/s) before launching each scene")
    parser.add_argument("--super_planning_horizon", type=float, default=None,
                        help="Override SUPER super_planner.planning_horizon (m) before launching each scene")
    parser.add_argument("--super_receding_dis", type=float, default=None,
                        help="Override SUPER super_planner.receding_dis (m) before launching each scene")
    parser.add_argument("--super_corridor_line_max_length", type=float, default=None,
                        help="Override SUPER super_planner.corridor_line_max_length (m) before launching each scene")
    parser.add_argument("--super_safe_corridor_line_max_length", type=float, default=None,
                        help="Override SUPER super_planner.safe_corridor_line_max_length (m) before launching each scene")
    parser.add_argument("--super_corridor_bound_dis", type=float, default=None,
                        help="Override SUPER super_planner.corridor_bound_dis (m) before launching each scene")
    parser.add_argument("--goal_reach_dist", type=float, default=2.0,
                        help="Treat trajectory as complete when current position is within this distance to goal")
    parser.add_argument("--goal_timeout", type=float, default=40.0,
                        help="Max seconds allowed for one goal execution across multiple replans")
    parser.add_argument("--max_replan_segments", type=int, default=20,
                        help="Max number of /planning_cmd/poly_traj segments to stitch for one goal")

    valid_keys = {a.dest for a in parser._actions}
    unknown = sorted(k for k in cfg if k not in valid_keys)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    parser.set_defaults(**cfg)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
