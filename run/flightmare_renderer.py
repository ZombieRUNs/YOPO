"""
Flightmare depth renderer node.

Subscribes to live odometry and drives Flightmare at render_hz (default 30 Hz),
publishing depth images for downstream FlowPilot inference.

ROS topics:
  SUB  <odom_topic>              (nav_msgs/Odometry) → drone pose/vel
  PUB  /flowpilot/depth          (sensor_msgs/Image, 32FC1, [0,1]) → depth
  PUB  /flowpilot/local_cloud    (PointCloud2)   → back-projected depth cloud
  PUB  /uav_mesh                 (Marker)        → drone marker
  PUB  /flowpilot/vel_marker     (Marker)        → speed text
  PUB  /local_map                (PointCloud2, latched) → scene point cloud
  PUB  /tf                                       → world → drone TF

Usage:
    conda activate yopo
    python flightmare_renderer.py \\
        --odom_topic /drone1_vrpn_client/estimated_odometry \\
        --scene_id 0 \\
        [--launch_unity] [--camera_pitch_deg -15]
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
from pathlib import Path
from typing import Optional

import numpy as np
from ruamel.yaml import YAML, RoundTripDumper, dump

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


# ---------------------------------------------------------------------------
# Shared odom state
# ---------------------------------------------------------------------------

_odom_lock = threading.Lock()
_odom_pos:  Optional[np.ndarray] = None
_odom_vel:  Optional[np.ndarray] = None
_odom_quat: Optional[np.ndarray] = None   # [w,x,y,z]

_flightmare_proc = None


def _callback_odom(msg: Odometry) -> None:
    global _odom_pos, _odom_vel, _odom_quat
    p = msg.pose.pose.position
    v = msg.twist.twist.linear
    q = msg.pose.pose.orientation
    with _odom_lock:
        _odom_pos  = np.array([p.x, p.y, p.z], dtype=np.float64)
        _odom_vel  = np.array([v.x, v.y, v.z], dtype=np.float64)
        _odom_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Flightmare process management
# ---------------------------------------------------------------------------

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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup_stale_flightmare(binary_name: str = 'flightmare.x86_64'):
    result = subprocess.run(['pgrep', '-f', binary_name],
                            capture_output=True, text=True, check=False)
    pids = [int(l.strip()) for l in result.stdout.splitlines()
            if l.strip() and l.strip().isdigit() and int(l.strip()) != os.getpid()]
    if not pids:
        return
    print(f'[INFO] terminating stale Flightmare: {pids}', flush=True)
    for pid in pids:
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not any(_pid_alive(p) for p in pids):
            return
        time.sleep(0.2)
    for pid in pids:
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass


def launch_flightmare(binary_path: str):
    cleanup_stale_flightmare(os.path.basename(binary_path))
    env  = os.environ.copy()
    if env.get('DISPLAY'):
        env.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
        env.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
        env.setdefault('__VK_LAYER_NV_optimus', 'NVIDIA_only')
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
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: pass


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _safe_quat(quat: np.ndarray) -> np.ndarray:
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = float(np.linalg.norm(quat))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (quat / n).astype(np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),   2*(x*z + w*y)  ],
        [2*(x*y + w*z),     1 - 2*(x*x+z*z), 2*(y*z - w*x)  ],
        [2*(x*z - w*y),     2*(y*z + w*x),   1 - 2*(x*x+y*y)],
    ], dtype=np.float64)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# ROS publish helpers
# ---------------------------------------------------------------------------

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
    r_ch = np.clip(255.0 * (2.0*t - 1.0),            0, 255).astype(np.uint8)
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
# PLY loader
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
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Flightmare renderer node: odom → depth image.')
    p.add_argument('--odom_topic',        type=str,
                   default='/some_object_name_vrpn_client/estimated_odometry')
    p.add_argument('--depth_topic',       type=str, default='/flowpilot/depth',
                   help='Published depth image topic consumed by inference node.')
    p.add_argument('--scene_id',          type=int,   default=0)
    p.add_argument('--spacing',           type=float, default=4.0)
    p.add_argument('--render_hz',         type=float, default=30.0)
    p.add_argument('--camera_pitch_deg',  type=float, default=-15.0)
    p.add_argument('--launch_unity',      action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    faulthandler.enable(all_threads=True)
    global _flightmare_proc

    args       = parse_args()
    script_dir = Path(__file__).resolve().parent

    # --- ROS init ---
    rospy.init_node('flightmare_renderer', anonymous=False)
    depth_pub       = rospy.Publisher(args.depth_topic,          Image,       queue_size=1)
    local_cloud_pub = rospy.Publisher('/flowpilot/local_cloud',  PointCloud2, queue_size=1)
    marker_pub      = rospy.Publisher('/uav_mesh',               Marker,      queue_size=1)
    vel_pub         = rospy.Publisher('/flowpilot/vel_marker',   Marker,      queue_size=1)
    map_pub         = rospy.Publisher('/local_map',              PointCloud2, queue_size=1, latch=True)
    tf_br           = tf2_ros.TransformBroadcaster()

    rospy.Subscriber(args.odom_topic, Odometry, _callback_odom, queue_size=1)

    ros_thread = threading.Thread(target=rospy.spin, daemon=True)
    ros_thread.start()
    print(f'[ROS] renderer started.  odom: {args.odom_topic}  depth→ {args.depth_topic}')

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
    cfg['env']['num_envs']    = 1
    cfg['env']['num_threads'] = 1
    cfg['env']['render']      = True
    cfg['env']['supervised']  = False
    cfg['env']['imitation']   = False
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env.connectUnity()
    env.setMapID(np.array([args.scene_id]))
    print(f'[env] scene {args.scene_id} ready.')

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
            print(f'[map] published {len(pts)} pts')
        except Exception as e:
            print(f'[map] failed: {e}')

    # Camera pitch quaternion
    pitch_rad = float(args.camera_pitch_deg) * math.pi / 180.0
    q_pitch   = np.array([math.cos(0.5*pitch_rad), 0.0,
                           math.sin(0.5*pitch_rad), 0.0], dtype=np.float64)

    render_dt = 1.0 / max(1e-3, float(args.render_hz))

    # --- Wait for first odometry ---
    print('[odom] waiting ...')
    while not rospy.is_shutdown():
        with _odom_lock:
            if _odom_pos is not None:
                break
        time.sleep(0.05)
    print(f'[odom] received. pos={_odom_pos.tolist()}')

    # --- 30 Hz render loop ---
    step_deadline = time.perf_counter()
    while not rospy.is_shutdown():
        with _odom_lock:
            cur_pos  = _odom_pos.copy()
            cur_vel  = _odom_vel.copy()
            cur_quat = _odom_quat.copy()

        # Render with odom quat (+ camera pitch)
        quat_render = _safe_quat(cur_quat)
        if abs(pitch_rad) > 1e-12:
            quat_render = _safe_quat(quat_mul(quat_render, q_pitch))

        env.setState(cur_pos, cur_vel, np.zeros(3), quat_render)
        env.render()
        depth_01 = env.getDepthImage()[0][0].astype(np.float32)

        publish_depth(depth_pub, depth_01)
        publish_drone_marker(marker_pub, cur_pos, quat_render)
        publish_vel_marker(vel_pub, cur_pos, cur_vel)
        broadcast_tf(tf_br, cur_pos, quat_render)
        local_msg = depth_to_local_cloud_msg(depth_01, cur_pos, quat_render)
        if local_msg is not None:
            local_cloud_pub.publish(local_msg)

        step_deadline += render_dt
        sleep_t = step_deadline - time.perf_counter()
        if sleep_t > 0.0:
            time.sleep(sleep_t)
        else:
            step_deadline = time.perf_counter()


if __name__ == '__main__':
    main()
