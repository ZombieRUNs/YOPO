"""
FlowPilot poly-action PX4-integrated test with RViz visualization.

Renders Flightmare at 30 Hz using live PX4/VRPN odometry.  Velocity and
position come from the odometry topic; acceleration and jerk are taken from
the nearest waypoint on the last FlowPilot-predicted trajectory.

Coordinate conventions (confirmed from ommpc_controller.hpp):
  - pose.pose.orientation  →  [w,x,y,z] quaternion, R_wb (world ← body)
  - twist.twist.linear     →  velocity already in world (ENU) frame
  - FlowPilot body frame   →  derived via differential flatness from vel/acc/jerk
                               (consistent with training data generation)
  - Rendering orientation  →  odometry quaternion (+ optional camera pitch)

ROS topics:
  SUB  /move_base_simple/goal                      → goal (RViz 2D Nav Goal)
  SUB  <odom_topic>  (nav_msgs/Odometry)           → drone state from PX4 / VRPN
  PUB  /flowpilot/best_traj_visual  (PointCloud2)  → predicted trajectory
  PUB  /depth_image                 (Image)         → current depth frame
  PUB  /uav_mesh                    (Marker)        → drone position marker
  PUB  /flowpilot/vel_marker        (Marker)        → speed text
  PUB  /flowpilot/goal_marker       (Marker)        → goal cylinder
  PUB  /flowpilot/local_cloud       (PointCloud2)   → back-projected depth cloud
  PUB  /tf                                          → world → drone TF

Usage:
    conda activate yopo
    python test_flowpilot_px4.py \\
        --flowpilot_root ../poly-action-rw \\
        --odom_topic /drone1_vrpn_client/estimated_odometry \\
        [--scene_id 0] [--compile_model]
"""

import argparse
import atexit
import faulthandler
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from math import comb as _comb
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from ruamel.yaml import YAML, RoundTripDumper, dump

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


# ---------------------------------------------------------------------------
# Shared state — written by ROS callbacks, read by main loop
# ---------------------------------------------------------------------------

_goal_lock = threading.Lock()
_goal_pos_world: Optional[np.ndarray] = None
_goal_received: bool = False
_goal_seq: int = 0

_odom_lock = threading.Lock()
_odom_pos:  Optional[np.ndarray] = None   # [3]    world-frame position
_odom_vel:  Optional[np.ndarray] = None   # [3]    world-frame velocity
# quaternion [w, x, y, z], R_wb (world ← body).
# Converted from ROS {x,y,z,w} to FlowPilot [w,x,y,z] in the callback.
_odom_quat: Optional[np.ndarray] = None

_flightmare_proc = None


def _callback_set_goal(msg: PoseStamped) -> None:
    global _goal_pos_world, _goal_received, _goal_seq
    with _goal_lock:
        _goal_pos_world = np.array([msg.pose.position.x,
                                    msg.pose.position.y,
                                    2.0], dtype=np.float64)
        _goal_received = True
        _goal_seq += 1
    print(f'[goal] new: ({_goal_pos_world[0]:.1f}, '
          f'{_goal_pos_world[1]:.1f}, 2.0)')


def _callback_odom(msg: Odometry) -> None:
    """Store latest odometry.  Velocity is expected in world (ENU) frame.

    Quaternion convention (from ommpc_controller.hpp feed()):
      pose.pose.orientation stores R_wb (world ← body) as a unit quaternion.
      ROS message layout: {x, y, z, w}  →  reorder to FlowPilot [w, x, y, z].
    """
    global _odom_pos, _odom_vel, _odom_quat
    p = msg.pose.pose.position
    v = msg.twist.twist.linear
    q = msg.pose.pose.orientation
    with _odom_lock:
        _odom_pos  = np.array([p.x, p.y, p.z], dtype=np.float64)
        _odom_vel  = np.array([v.x, v.y, v.z], dtype=np.float64)
        # ROS stores quaternion as {x,y,z,w}; FlowPilot uses [w,x,y,z]
        _odom_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Flightmare process management (verbatim from test_flowpilot_rviz.py)
# ---------------------------------------------------------------------------

def build_flightmare_env():
    env = os.environ.copy()
    if env.get('DISPLAY'):
        env.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
        env.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
        env.setdefault('__VK_LAYER_NV_optimus', 'NVIDIA_only')
    return env


def cleanup_stale_flightmare(binary_name: str = 'flightmare.x86_64'):
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
    cleanup_stale_flightmare(os.path.basename(binary_path))
    env  = build_flightmare_env()
    stub = _build_alsa_stub()
    if stub:
        preload = env.get('LD_PRELOAD', '')
        env['LD_PRELOAD'] = f'{stub}:{preload}' if preload else stub
    proc = subprocess.Popen(
        [binary_path],
        cwd=os.path.dirname(binary_path),
        env=env,
        start_new_session=True)
    return proc


def terminate_process(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Geometry helpers (verbatim from test_flowpilot_rviz.py)
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    pos:  np.ndarray   # [3] world frame
    vel:  np.ndarray   # [3] world frame
    acc:  np.ndarray   # [3] world frame
    jerk: np.ndarray   # [3] world frame


def _safe_quat(quat: np.ndarray) -> np.ndarray:
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = float(np.linalg.norm(quat))
    if n < 1e-9 or not np.isfinite(n):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (quat / n).astype(np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[qw,qx,qy,qz] → R_wb (world ← body).  v_world = R @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),   2*(x*z + w*y)  ],
        [2*(x*y + w*z),     1 - 2*(x*x+z*z), 2*(y*z - w*x)  ],
        [2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x+y*y)],
    ], dtype=np.float64)


def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def yaw_frame_from_start_to_goal(start, goal, eps=1e-6):
    dx, dy = float(goal[0]-start[0]), float(goal[1]-start[1])
    if dx*dx + dy*dy < eps*eps:
        return None, None
    yaw = float(np.arctan2(dy, dx))
    cy, sy = float(np.cos(0.5*yaw)), float(np.sin(0.5*yaw))
    q = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
    return q, quat_to_rot(q)


def build_state_goal(cur: WorldState, goal_pos_world: np.ndarray,
                     get_body_frame_fn,
                     quat_wb_override=None, r_wb_override=None):
    """Build FlowPilot state_goal [25] in body frame derived from flatness.

    The body frame is always computed via differential flatness (vel/acc/jerk),
    consistent with how the training dataset was generated.  The odometry
    quaternion is NOT used here — it is used only for Flightmare rendering.
    """
    if quat_wb_override is not None:
        quat_wb = quat_wb_override.astype(np.float64)
        r_wb    = r_wb_override.astype(np.float64)
    else:
        quat_wb, r_wb = get_body_frame_fn(cur.vel, cur.acc, cur.jerk)
    r_bw       = r_wb.T
    state_pos  = np.zeros(3, dtype=np.float32)
    state_vel  = (r_bw @ cur.vel ).astype(np.float32)
    state_acc  = (r_bw @ cur.acc ).astype(np.float32)
    state_jerk = (r_bw @ cur.jerk).astype(np.float32)
    goal_pos_b = (r_bw @ (goal_pos_world - cur.pos)).astype(np.float32)
    goal_vel_b = np.zeros(3, dtype=np.float32)
    goal_acc_b = np.zeros(3, dtype=np.float32)
    state_goal = np.concatenate([state_pos, state_vel, state_acc, state_jerk,
                                  quat_wb.astype(np.float32),
                                  goal_pos_b, goal_vel_b, goal_acc_b])
    return state_goal, quat_wb, r_wb


# ---------------------------------------------------------------------------
# Reference trajectory helpers
# ---------------------------------------------------------------------------

def find_nearest_wp(wp_world: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Return the row of wp_world [N,12] whose position is closest to pos [3].

    Used to extract reference acc/jerk when real dynamics (PX4) provide pos/vel.
    """
    dists = np.linalg.norm(wp_world[:, 0:3] - pos[None], axis=1)
    return wp_world[int(np.argmin(dists))]


# ---------------------------------------------------------------------------
# ROS publish helpers (verbatim from test_flowpilot_rviz.py)
# ---------------------------------------------------------------------------

def publish_trajectory(pub: rospy.Publisher, wp_world: np.ndarray,
                       frame_id: str = 'world'):
    pts = wp_world[:, 0:3].astype(np.float32)
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    pub.publish(point_cloud2.create_cloud_xyz32(header, pts))


def publish_depth(pub: rospy.Publisher, depth_01: np.ndarray,
                  frame_id: str = 'camera'):
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height, msg.width = depth_01.shape
    msg.encoding = '32FC1'
    msg.step = msg.width * 4
    msg.data = depth_01.astype(np.float32).tobytes()
    pub.publish(msg)


def publish_drone_marker(pub: rospy.Publisher, pos: np.ndarray,
                         quat_wxyz: np.ndarray, frame_id: str = 'world'):
    m = Marker()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = frame_id
    m.ns = 'drone'; m.id = 0
    m.type = Marker.SPHERE; m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2])
    m.pose.orientation.w = float(quat_wxyz[0])
    m.pose.orientation.x = float(quat_wxyz[1])
    m.pose.orientation.y = float(quat_wxyz[2])
    m.pose.orientation.z = float(quat_wxyz[3])
    m.scale.x = m.scale.y = m.scale.z = 0.4
    m.color.r = 0.2; m.color.g = 0.6; m.color.b = 1.0; m.color.a = 1.0
    pub.publish(m)


def publish_vel_marker(pub: rospy.Publisher, pos: np.ndarray,
                       vel: np.ndarray, frame_id: str = 'world'):
    speed = float(np.linalg.norm(vel))
    m = Marker()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = frame_id
    m.ns = 'velocity'; m.id = 2
    m.type = Marker.TEXT_VIEW_FACING; m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2]) + 5.5
    m.pose.orientation.w = 1.0
    m.scale.z = 3.0
    m.text = f'{speed:.1f} m/s'
    m.color.r = 0.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0
    pub.publish(m)


def publish_goal_marker(pub: rospy.Publisher, goal: np.ndarray,
                        frame_id: str = 'world'):
    m = Marker()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = frame_id
    m.ns = 'goal'; m.id = 1
    m.type = Marker.CYLINDER; m.action = Marker.ADD
    m.pose.position.x = float(goal[0])
    m.pose.position.y = float(goal[1])
    m.pose.position.z = float(goal[2])
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = 0.8; m.scale.z = 4.0
    m.color.r = 1.0; m.color.g = 0.4; m.color.b = 0.0; m.color.a = 0.6
    pub.publish(m)


def broadcast_tf(br: tf2_ros.TransformBroadcaster,
                 pos: np.ndarray, quat_wxyz: np.ndarray,
                 child: str = 'drone', parent: str = 'world'):
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x = float(pos[0])
    t.transform.translation.y = float(pos[1])
    t.transform.translation.z = float(pos[2])
    t.transform.rotation.w = float(quat_wxyz[0])
    t.transform.rotation.x = float(quat_wxyz[1])
    t.transform.rotation.y = float(quat_wxyz[2])
    t.transform.rotation.z = float(quat_wxyz[3])
    br.sendTransform(t)


# ---------------------------------------------------------------------------
# Depth → local point cloud (verbatim from test_flowpilot_rviz.py)
# ---------------------------------------------------------------------------

_CAM_FX: float = 80.0
_CAM_FY: float = 80.0
_CAM_CX: float = 79.5
_CAM_CY: float = 47.5
_CAM_MAX_M: float = 20.0


def depth_to_local_cloud_msg(depth_01: np.ndarray, pos: np.ndarray,
                              quat_wxyz: np.ndarray,
                              vis_max_m: float = 8.0,
                              skip: int = 3,
                              frame_id: str = 'world'):
    H, W = depth_01.shape
    us = np.arange(0, W, skip, dtype=np.int32)
    vs = np.arange(0, H, skip, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()
    dd = depth_01[vv, uu] * _CAM_MAX_M
    valid = (dd > 0.2) & (dd < vis_max_m) & np.isfinite(dd)
    uu, vv, dd = uu[valid], vv[valid], dd[valid]
    if len(dd) == 0:
        return None
    x_c = (uu.astype(np.float32) - _CAM_CX) / _CAM_FX * dd
    y_c = (vv.astype(np.float32) - _CAM_CY) / _CAM_FY * dd
    pts_body = np.stack([dd, -x_c, -y_c], axis=1)
    R_wb     = quat_to_rot(quat_wxyz)
    pts_w    = (R_wb @ pts_body.T).T + pos
    t = np.clip(dd / vis_max_m, 0.0, 1.0)
    r_ch = np.clip(255.0 * (2.0*t - 1.0),           0, 255).astype(np.uint8)
    g_ch = np.clip(255.0 * (1.0 - np.abs(2.0*t-1.0)), 0, 255).astype(np.uint8)
    b_ch = np.clip(255.0 * (1.0 - 2.0*t),             0, 255).astype(np.uint8)
    rgb_u32 = (r_ch.astype(np.uint32) << 16
               | g_ch.astype(np.uint32) << 8
               | b_ch.astype(np.uint32))
    N = len(dd)
    buf = np.zeros(N, dtype=[('x', np.float32), ('y', np.float32),
                              ('z', np.float32), ('rgb', np.float32)])
    buf['x'] = pts_w[:, 0].astype(np.float32)
    buf['y'] = pts_w[:, 1].astype(np.float32)
    buf['z'] = pts_w[:, 2].astype(np.float32)
    buf['rgb'] = rgb_u32.view(np.float32)
    fields = [
        PointField('x',   0, PointField.FLOAT32, 1),
        PointField('y',   4, PointField.FLOAT32, 1),
        PointField('z',   8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1),
    ]
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    msg = PointCloud2()
    msg.header      = header
    msg.height      = 1
    msg.width       = N
    msg.fields      = fields
    msg.is_bigendian = False
    msg.point_step  = 16
    msg.row_step    = 16 * N
    msg.data        = buf.tobytes()
    msg.is_dense    = True
    return msg


# ---------------------------------------------------------------------------
# PLY loader + voxel downsample (verbatim from test_flowpilot_rviz.py)
# ---------------------------------------------------------------------------

def _voxel_downsample(pts: np.ndarray, voxel_size: float = 0.3) -> np.ndarray:
    if len(pts) == 0:
        return pts
    idx = np.floor(pts / voxel_size).astype(np.int32)
    mn = idx.min(axis=0)
    idx -= mn
    mx = idx.max(axis=0) + 1
    keys = (idx[:, 0].astype(np.int64) * int(mx[1]) * int(mx[2])
            + idx[:, 1].astype(np.int64) * int(mx[2])
            + idx[:, 2].astype(np.int64))
    _, first = np.unique(keys, return_index=True)
    return pts[first]


def _load_ply_xyz(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        header   = '\n'.join(header_lines)
        n_verts  = 0
        for line in header_lines:
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
                break
        is_binary_le = 'format binary_little_endian' in header
        is_binary_be = 'format binary_big_endian' in header
        props, in_vertex = [], False
        for line in header_lines:
            if line.startswith('element vertex'):
                in_vertex = True
            elif line.startswith('element') and in_vertex:
                break
            elif in_vertex and line.startswith('property'):
                parts = line.split()
                props.append((parts[1], parts[2]))
        prop_names = [p[1] for p in props]
        prop_types = [p[0] for p in props]
        type_map   = {'float': 'f4', 'float32': 'f4', 'double': 'f8',
                      'int': 'i4', 'uint': 'u4', 'uchar': 'u1', 'short': 'i2'}
        dtype = np.dtype([(n, type_map.get(t, 'f4'))
                          for n, t in zip(prop_names, prop_types)])
        if is_binary_le or is_binary_be:
            raw = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)
            if is_binary_be:
                raw = raw.byteswap().newbyteorder()
        else:
            rows = [f.readline().decode().split() for _ in range(n_verts)]
            raw  = np.array([tuple(r) for r in rows], dtype=dtype)
    return np.stack([raw['x'].astype(np.float32),
                     raw['y'].astype(np.float32),
                     raw['z'].astype(np.float32)], axis=1)


# ---------------------------------------------------------------------------
# FlowPilot inference (verbatim from test_flowpilot_rviz.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_one_step(model, vae_encoder, basis, scheduler,
                   cond_depth, state_goal, sg_mean, sg_std,
                   free_mean, free_std, v_cmd, num_steps, device):
    """Returns (wp_body [48,12], bernstein_ctrl_pts_body [8,3]).

    bernstein_ctrl_pts_body are the 8 Bernstein control points in body frame
    (physical units, metres).  Use these to reconstruct the polynomial for
    publishing to OMMPC after transforming to world frame.
    """
    cond_frame   = torch.from_numpy(cond_depth.astype(np.float32)
                                    ).unsqueeze(0).unsqueeze(0).to(device)
    sg_t         = torch.from_numpy(state_goal.astype(np.float32)
                                    ).unsqueeze(0).to(device)
    sg_norm      = (sg_t - sg_mean) / sg_std
    init_pva     = sg_t[:, 0:9]
    constrained  = basis.constrained_coeffs(init_pva)
    v_cmd_t      = torch.tensor([v_cmd], dtype=torch.float32, device=device)

    cond_latent  = vae_encoder.encode(cond_frame.unsqueeze(1))
    video_latent = torch.randn(1, 48, 3, 6, 10, device=device)
    video_latent[:, :, 0:1] = cond_latent
    free_latent  = torch.randn(1, 5, 3, device=device)

    sigmas   = scheduler.inference_sigmas(num_steps).to(device)
    t_linear = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_ids   = (t_linear[i] * 999).long().clamp(0, 999).unsqueeze(0)
        d_sigma = sigmas[i + 1] - sigmas[i]
        v_pred, c_pred = model(video_latent, sg_norm, v_cmd_t,
                               free_latent, t_ids, t_ids)
        video_latent = video_latent.clone()
        video_latent[:, :, 1:] = video_latent[:, :, 1:] + v_pred * d_sigma
        video_latent[:, :, 0:1] = cond_latent
        free_latent = free_latent + c_pred * d_sigma

    free_denorm = free_latent * free_std + free_mean
    wp_body     = basis.expand_to_12d(free_denorm, constrained)[0]

    # All 8 Bernstein control points in body frame [8, 3]
    all_ctrl_pts = torch.cat([constrained, free_denorm], dim=1)[0]  # [8, 3]

    return (wp_body.cpu().numpy().astype(np.float32),
            all_ctrl_pts.cpu().numpy().astype(np.float64))


# ---------------------------------------------------------------------------
# Bernstein → power basis conversion for OMMPC (traj_utils/PolyTraj)
# ---------------------------------------------------------------------------

def _bernstein_to_power_descending(b_pts: np.ndarray, T: float) -> np.ndarray:
    """Convert degree-7 Bernstein control points to power basis (descending order).

    OMMPC polynomial_trajectory.h evaluates:
        p(t) = col(0)*t^n + col(1)*t^(n-1) + ... + col(n)*t^0
    so coef[0] = highest-degree coefficient.

    Args:
        b_pts: [8, 3] Bernstein control points, world frame, physical units (metres).
        T:     trajectory duration in seconds.

    Returns:
        coefs: [8, 3] descending power basis.
               coefs[0]*t^7 + coefs[1]*t^6 + ... + coefs[7] = p(t)
    """
    n = len(b_pts) - 1   # 7
    # Bernstein-to-power matrix in τ = t/T domain:
    #   c_tau[j] = Σᵢ M[j,i] * b[i]
    #   M[j,i] = C(n,i) * C(n-i, j-i) * (-1)^(j-i)   for j >= i, else 0
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    for j in range(n + 1):
        for i in range(j + 1):
            M[j, i] = (_comb(n, i) * _comb(n - i, j - i)
                       * ((-1) ** (j - i)))
    c_tau = M @ b_pts.astype(np.float64)    # [8, 3], coeff of τ^j (ascending)

    # Rescale τ → t:  a_j = c_tau[j] / T^j  is the coefficient of t^j
    a_t = np.array([c_tau[j] / (T ** j) for j in range(n + 1)])  # [8,3] ascending

    # Reverse to descending (OMMPC convention)
    return a_t[::-1].copy()   # [8, 3]


def publish_poly_traj(pub, coefs_world: np.ndarray, T: float,
                      traj_id: int, start_time) -> None:
    """Publish traj_utils/PolyTraj for OMMPC to track.

    Args:
        coefs_world: [8, 3] descending power basis (world frame).
        T:           trajectory duration in seconds.
        traj_id:     monotonically increasing integer (must start from 1).
        start_time:  rospy.Time when the trajectory is valid from.
    """
    from traj_utils.msg import PolyTraj
    msg          = PolyTraj()
    msg.drone_id = 0
    msg.traj_id  = int(traj_id)
    msg.start_time = start_time
    msg.order    = 7
    msg.coef_x   = coefs_world[:, 0].astype(np.float32).tolist()
    msg.coef_y   = coefs_world[:, 1].astype(np.float32).tolist()
    msg.coef_z   = coefs_world[:, 2].astype(np.float32).tolist()
    msg.duration = [float(T)]
    pub.publish(msg)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='FlowPilot PX4-integrated test (odometry-driven rendering).')
    p.add_argument('--flowpilot_root',     type=str,
                   default='/root/workspace/YOPO/run/poly-action-rw')
    p.add_argument('--device',             type=str, default='cuda')
    p.add_argument('--phase2_ckpt',        type=str,
                   default='/root/workspace/YOPO/run/poly-action-rw/ckpt/super/ckpt_final.pt')
    p.add_argument('--vae_pth',            type=str,
                   default='/root/workspace/YOPO/run/poly-action-rw/ckpt/2026-3-24/Wan2.2_VAE.pth')
    p.add_argument('--depth_encoder_ckpt', type=str, default='',
                   help='If set, use lightweight DepthEncoder instead of Wan2.2 VAE.')
    p.add_argument('--action_stats_path',  type=str,
                   default='/root/workspace/YOPO/run/poly-action-rw/ckpt/super/action_stats_super.pt')
    p.add_argument('--coeff_stats_path',   type=str,
                   default='/root/workspace/YOPO/run/poly-action-rw/ckpt/super/coeff_stats_super.pt')
    p.add_argument('--odom_topic',         type=str,
                   default='/some_object_name_vrpn_client/estimated_odometry',
                   help='nav_msgs/Odometry topic from PX4/VRPN ground-truth.')
    p.add_argument('--scene_id',           type=int, default=0)
    p.add_argument('--spacing',            type=float, default=4.0)
    p.add_argument('--num_steps',          type=int, default=5)
    p.add_argument('--infer_every_n',      type=int, default=24,
                   help='Re-run inference every N render frames (30 Hz → default 0.8 s).')
    p.add_argument('--v_cmd',             type=float, default=6.0)
    p.add_argument('--goal_reach_dist',   type=float, default=2.0)
    p.add_argument('--camera_pitch_deg',  type=float, default=-15.0)
    p.add_argument('--render_hz',         type=float, default=30.0)
    p.add_argument('--launch_unity',      action='store_true')
    p.add_argument('--compile_model',     action='store_true')
    p.add_argument('--num_layers',        type=int, default=12)
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    faulthandler.enable(all_threads=True)
    global _flightmare_proc

    args       = parse_args()
    script_dir = Path(__file__).resolve().parent

    def ckpt(path):
        return path if os.path.isabs(path) else str(script_dir / path)

    # --- ROS init ---
    rospy.init_node('flowpilot_px4', anonymous=False)
    traj_pub        = rospy.Publisher('/flowpilot/best_traj_visual', PointCloud2, queue_size=1)
    map_pub         = rospy.Publisher('/local_map',                  PointCloud2, queue_size=1, latch=True)
    local_cloud_pub = rospy.Publisher('/flowpilot/local_cloud',      PointCloud2, queue_size=1)
    depth_pub       = rospy.Publisher('/depth_image',                Image,       queue_size=1)
    marker_pub      = rospy.Publisher('/uav_mesh',                   Marker,      queue_size=1)
    vel_pub         = rospy.Publisher('/flowpilot/vel_marker',       Marker,      queue_size=1)
    goal_pub        = rospy.Publisher('/flowpilot/goal_marker',      Marker,      queue_size=1)
    tf_br           = tf2_ros.TransformBroadcaster()

    rospy.Subscriber('/move_base_simple/goal', PoseStamped, _callback_set_goal, queue_size=1)
    rospy.Subscriber(args.odom_topic, Odometry, _callback_odom, queue_size=1)

    ros_thread = threading.Thread(target=rospy.spin, daemon=True)
    ros_thread.start()
    print(f'[ROS] node started.  Odometry: {args.odom_topic}')

    # --- FlowPilot imports ---
    flowpilot_root = ckpt(args.flowpilot_root)
    if flowpilot_root not in sys.path:
        sys.path.insert(0, flowpilot_root)
    from flowpilot.scheduler import FlowMatchScheduler
    from flowpilot.models.flowpilot import FlowPilotPhase2
    from flowpilot.models.vae_encoder import WanVAEEncoder
    from flowpilot.models.depth_encoder import DepthEncoder
    from flowpilot.models.poly_action import BernsteinActionBasis
    from flowpilot.coords import body_to_world, get_body_frame

    # traj_utils requires the catkin devel workspace to be sourced
    from traj_utils.msg import PolyTraj
    poly_traj_pub = rospy.Publisher('/drone_0_planning/trajectory',
                                    PolyTraj, queue_size=1)

    # --- Flightmare env ---
    flightmare_path = os.environ.get('FLIGHTMARE_PATH', str(script_dir.parent))
    os.environ.setdefault('FLIGHTMARE_PATH', flightmare_path)
    from flightgym import QuadrotorEnv_v1
    from flightpolicy.envs import vec_env_wrapper as wrapper

    if args.launch_unity:
        binary_path = os.path.join(
            flightmare_path, 'flightrender/RPG_Flightmare/flightmare.x86_64')
        _flightmare_proc = launch_flightmare(binary_path)
        atexit.register(terminate_process, _flightmare_proc)

    cfg_path = os.path.join(flightmare_path, 'flightlib', 'configs', 'vec_env.yaml')
    cfg = YAML().load(open(cfg_path))
    cfg['env']['num_envs']   = 1
    cfg['env']['num_threads'] = 1
    cfg['env']['render']     = True
    cfg['env']['supervised'] = False
    cfg['env']['imitation']  = False
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env.connectUnity()
    env.setMapID(np.array([args.scene_id]))
    wb = env.world_box.astype(np.float64)
    print(f'[env] scene {args.scene_id} ready, world_box={wb.tolist()}')

    env.spawnTreesAndSavePointcloud(args.scene_id, spacing=args.spacing)
    env.reset()
    env.render()

    # Publish scene point cloud (latched)
    ply_path = os.path.join(flightmare_path, 'run', 'yopo_sim',
                            f'pointcloud-{args.scene_id}.ply')
    if os.path.exists(ply_path):
        try:
            pts = _voxel_downsample(_load_ply_xyz(ply_path), voxel_size=0.3)
            h = Header(); h.stamp = rospy.Time.now(); h.frame_id = 'world'
            map_pub.publish(point_cloud2.create_cloud_xyz32(h, pts))
            print(f'[map] published {len(pts)} pts from {ply_path}')
        except Exception as e:
            print(f'[map] failed: {e}')

    # --- Model ---
    model = FlowPilotPhase2(
        num_layers=args.num_layers, d_video=512, d_action=256, num_heads=8,
        ffn_dim_video=2048, ffn_dim_action=1024, z_dim=48,
        num_actions=5, latent_t=3, grid_h=6, grid_w=10,
    ).to(args.device)
    ckpt_d = torch.load(ckpt(args.phase2_ckpt), map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt_d['model'])
    model.eval()
    if args.compile_model:
        model = torch.compile(model, mode='reduce-overhead')
        print('[model] compiled with torch.compile(mode=reduce-overhead)')

    if args.depth_encoder_ckpt:
        vae_encoder = DepthEncoder(z_dim=48).to(args.device).eval()
        _de_ckpt = torch.load(args.depth_encoder_ckpt, map_location=args.device, weights_only=True)
        vae_encoder.load_state_dict(_de_ckpt['model'])
        print(f'[vae] depth_encoder loaded ({args.depth_encoder_ckpt})')
    else:
        vae_encoder = WanVAEEncoder(ckpt(args.vae_pth),
                                    dtype=torch.bfloat16, device=args.device)
        vae_encoder.eval()

    basis     = BernsteinActionBasis(degree=7, T=1.6, num_waypoints=48, hz=30,
                                     device=args.device)
    scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    astats   = torch.load(ckpt(args.action_stats_path), map_location=args.device, weights_only=True)
    cstats   = torch.load(ckpt(args.coeff_stats_path),  map_location=args.device, weights_only=True)
    sg_mean  = astats['sg_mean'].to(args.device)
    sg_std   = astats['sg_std'].to(args.device)
    free_mean = cstats['free_mean'].to(args.device)
    free_std  = cstats['free_std'].to(args.device)
    print('[model] weights loaded.')

    # Warm-up
    torch.set_grad_enabled(False)
    _dummy_depth = np.zeros((96, 160), dtype=np.float32)
    _dummy_sg    = np.zeros(25, dtype=np.float32)
    infer_one_step(model, vae_encoder, basis, scheduler,
                   _dummy_depth, _dummy_sg, sg_mean, sg_std,
                   free_mean, free_std, args.v_cmd, args.num_steps, args.device)
    print('[model] warm-up done.')

    # Camera pitch quaternion (applied on top of body quat for rendering)
    pitch_rad = float(args.camera_pitch_deg) * math.pi / 180.0
    q_pitch   = np.array([math.cos(0.5*pitch_rad), 0.0,
                           math.sin(0.5*pitch_rad), 0.0], dtype=np.float64)

    render_dt = 1.0 / max(1e-3, float(args.render_hz))

    # --- Wait for first odometry ---
    print('[odom] waiting for first odometry message ...')
    while not rospy.is_shutdown():
        with _odom_lock:
            if _odom_pos is not None:
                break
        time.sleep(0.05)
    print(f'[odom] received.  pos={_odom_pos.tolist()}')

    # --- Single flat 30 Hz render + inference loop ---
    # Every frame: snapshot odometry → render → publish.
    # Every infer_every_n frames: re-run FlowPilot inference → update wp_world.
    # There is no inner loop.  The drone moves under real PX4 dynamics; we just
    # follow its odometry.

    print('[RViz] set goal with 2D Nav Goal.')
    wp_world: Optional[np.ndarray] = None   # [N,12] last predicted traj, world frame
    goal_pos: Optional[np.ndarray] = None
    goal_seq    = -1
    render_step = 0
    traj_id     = 0   # monotonically increasing; OMMPC requires traj_id >= 1
    step_deadline = time.perf_counter()
    in_hover = False

    while not rospy.is_shutdown():

        # ---- Check / refresh goal -------------------------------------------
        with _goal_lock:
            goal_ready = _goal_received
            new_goal   = _goal_pos_world.copy() if _goal_received else None
            new_seq    = _goal_seq

        if not goal_ready:
            time.sleep(0.05)
            continue

        if goal_pos is None or new_seq != goal_seq:
            goal_pos    = new_goal
            goal_seq    = new_seq
            wp_world    = None
            render_step = 0
            in_hover    = False
            print(f'[run] goal=({goal_pos[0]:.1f},{goal_pos[1]:.1f},{goal_pos[2]:.1f})')

        # ---- Snapshot latest odometry ---------------------------------------
        with _odom_lock:
            cur_pos  = _odom_pos.copy()
            cur_vel  = _odom_vel.copy()
            cur_quat = _odom_quat.copy()   # [w,x,y,z] R_wb

        # ---- ref acc/jerk: nearest point on last predicted trajectory -------
        ref_acc  = np.zeros(3, dtype=np.float64)
        ref_jerk = np.zeros(3, dtype=np.float64)
        if wp_world is not None:
            nearest  = find_nearest_wp(wp_world, cur_pos)
            ref_acc  = nearest[6:9].astype(np.float64)
            ref_jerk = nearest[9:12].astype(np.float64)

        # ---- Render using flatness-derived orientation (consistent with training) ----
        # Policy was trained with yaw = velocity direction (via get_body_frame).
        # Use that same convention for rendering so depth images match training distribution.
        # Position comes from odometry; orientation comes from differential flatness.
        quat_flat, _ = get_body_frame(cur_vel, ref_acc, ref_jerk)
        quat_render = _safe_quat(np.asarray(quat_flat, dtype=np.float64))
        if abs(pitch_rad) > 1e-12:
            quat_render = _safe_quat(quat_mul(quat_render, q_pitch))

        env.setState(cur_pos, cur_vel, ref_acc, quat_render)
        env.render()
        cond_depth = env.getDepthImage()[0][0].astype(np.float32)

        # ---- Publish every frame --------------------------------------------
        publish_depth(depth_pub, cond_depth)
        publish_drone_marker(marker_pub, cur_pos, quat_render)
        publish_vel_marker(vel_pub, cur_pos, cur_vel)
        broadcast_tf(tf_br, cur_pos, quat_render)
        local_msg = depth_to_local_cloud_msg(cond_depth, cur_pos, quat_render)
        if local_msg is not None:
            local_cloud_pub.publish(local_msg)
        if goal_pos is not None:
            publish_goal_marker(goal_pub, goal_pos)

        # ---- Periodic inference (every infer_every_n frames) ----------------
        if not in_hover and render_step % args.infer_every_n == 0:
            cur_state = WorldState(pos=cur_pos, vel=cur_vel,
                                   acc=ref_acc, jerk=ref_jerk)

            # First inference: yaw body frame toward goal
            init_q, init_r = yaw_frame_from_start_to_goal(cur_pos, goal_pos)
            if wp_world is None and init_q is not None:
                state_goal, quat_wb, r_wb = build_state_goal(
                    cur_state, goal_pos, get_body_frame,
                    quat_wb_override=init_q, r_wb_override=init_r)
            else:
                state_goal, quat_wb, r_wb = build_state_goal(
                    cur_state, goal_pos, get_body_frame)

            t0 = time.perf_counter()
            wp_body, ctrl_pts_body = infer_one_step(
                model, vae_encoder, basis, scheduler,
                cond_depth, state_goal, sg_mean, sg_std,
                free_mean, free_std, args.v_cmd, args.num_steps, args.device)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            if np.all(np.isfinite(wp_body)) and np.all(np.isfinite(ctrl_pts_body)):
                # body frame → world frame (flatness-derived R_wb, consistent with training)
                wp_new = body_to_world(wp_body, cur_pos, r_wb).astype(np.float64)
                wp_new[:, 0] = np.clip(wp_new[:, 0], wb[0]+0.2, wb[3]-0.2)
                wp_new[:, 1] = np.clip(wp_new[:, 1], wb[1]+0.2, wb[4]-0.2)
                wp_new[:, 2] = np.clip(wp_new[:, 2], max(0.05, wb[2]+0.05), wb[5]-0.1)
                wp_world = wp_new
                publish_trajectory(traj_pub, wp_world)

                # Bernstein control points: body → world, then → power basis → OMMPC
                ctrl_pts_world = (r_wb @ ctrl_pts_body.T).T + cur_pos  # [8, 3]
                coefs_world = _bernstein_to_power_descending(ctrl_pts_world, basis.T)
                traj_id += 1
                publish_poly_traj(poly_traj_pub, coefs_world, basis.T,
                                  traj_id, rospy.Time.now())
            else:
                print('[run] NaN/Inf in inference output, keeping previous trajectory.')

            dist = float(np.linalg.norm(goal_pos - cur_pos))
            print(f'[run] step={render_step}  infer={infer_ms:.1f}ms  '
                  f'dist={dist:.2f}m  '
                  f'pos=({cur_pos[0]:.1f},{cur_pos[1]:.1f},{cur_pos[2]:.1f})')

        # ---- Goal-reached check → hover mode --------------------------------
        if not in_hover:
            dist = float(np.linalg.norm(goal_pos - cur_pos))
            if dist < args.goal_reach_dist:
                print(f'[run] arrived! dist={dist:.2f}m. Hovering — send new goal to continue.')
                in_hover = True

        render_step += 1

        # ---- Real-time pacing at render_hz ----------------------------------
        step_deadline += render_dt
        sleep_t = step_deadline - time.perf_counter()
        if sleep_t > 0.0:
            time.sleep(sleep_t)
        else:
            # fell behind (e.g. inference took too long); reset deadline
            step_deadline = time.perf_counter()


if __name__ == '__main__':
    main()
