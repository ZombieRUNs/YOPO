"""
FlowPilot poly-action closed-loop env test with RViz visualization.

Inference + Flightmare rendering loop (test_flowpilot_env_loop.py logic), with
ROS topics for RViz:
  SUB  /move_base_simple/goal        → goal position (set via RViz 2D Nav Goal)
  PUB  /flowpilot/best_traj_visual   → planned trajectory (PointCloud2, world frame)
  PUB  /depth_image                  → current depth frame (sensor_msgs/Image)
  PUB  /uav_mesh                     → drone position marker (visualization_msgs/Marker)
  PUB  /tf                           → world→drone TF for RViz frame tracking

Usage:
    conda activate yopo
    python test_flowpilot_rviz.py --flowpilot_root ../poly-action-rw [--compile_model] [--scene_id 0]
    # then in RViz: set Fixed Frame = world, add topics above, click 2D Nav Goal to set destination
"""

import argparse
import faulthandler
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ruamel.yaml import YAML, RoundTripDumper, dump

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


# ---------------------------------------------------------------------------
# Shared state (written by ROS goal callback, read by main loop)
# ---------------------------------------------------------------------------

_goal_lock = threading.Lock()
_goal_pos_world: np.ndarray = None   # [3] set when first goal arrives
_goal_received: bool = False


def _callback_set_goal(msg: PoseStamped):
    global _goal_pos_world, _goal_received
    with _goal_lock:
        _goal_pos_world = np.array([msg.pose.position.x, msg.pose.position.y, 2.0])
        _goal_received = True
    print(f'[RViz] New goal: ({_goal_pos_world[0]:.1f}, {_goal_pos_world[1]:.1f}, 2.0)')


# ---------------------------------------------------------------------------
# ROS publish helpers
# ---------------------------------------------------------------------------

def publish_trajectory(pub: rospy.Publisher, wp_world: np.ndarray, frame_id: str = 'world'):
    """Publish trajectory waypoints xyz as PointCloud2."""
    pts = wp_world[:, 0:3].astype(np.float32)
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    pub.publish(point_cloud2.create_cloud_xyz32(header, pts))


def publish_depth(pub: rospy.Publisher, depth_01: np.ndarray, frame_id: str = 'camera'):
    """Publish normalized [0,1] float32 depth as sensor_msgs/Image (32FC1)."""
    msg = Image()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height, msg.width = depth_01.shape
    msg.encoding = '32FC1'
    msg.step = msg.width * 4
    msg.data = depth_01.astype(np.float32).tobytes()
    pub.publish(msg)


def publish_drone_marker(pub: rospy.Publisher, pos: np.ndarray, quat_wxyz: np.ndarray,
                         frame_id: str = 'world'):
    """Publish a sphere marker at drone position."""
    m = Marker()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = frame_id
    m.ns = 'drone'
    m.id = 0
    m.type = Marker.SPHERE
    m.action = Marker.ADD
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


def publish_goal_marker(pub: rospy.Publisher, goal: np.ndarray, frame_id: str = 'world'):
    """Publish a cylinder marker at goal position."""
    m = Marker()
    m.header.stamp = rospy.Time.now()
    m.header.frame_id = frame_id
    m.ns = 'goal'
    m.id = 1
    m.type = Marker.CYLINDER
    m.action = Marker.ADD
    m.pose.position.x = float(goal[0])
    m.pose.position.y = float(goal[1])
    m.pose.position.z = float(goal[2])
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = 0.8; m.scale.z = 4.0
    m.color.r = 1.0; m.color.g = 0.4; m.color.b = 0.0; m.color.a = 0.6
    pub.publish(m)


def broadcast_tf(br: tf2_ros.TransformBroadcaster, pos: np.ndarray,
                 quat_wxyz: np.ndarray, child: str = 'drone', parent: str = 'world'):
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
# Geometry helpers (from test_flowpilot_env_loop.py)
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    jerk: np.ndarray


def _safe_quat(quat: np.ndarray) -> np.ndarray:
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = float(np.linalg.norm(quat))
    if n < 1e-9 or not np.isfinite(n):
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
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def obs_to_world_pva(obs):
    pos  = obs[0, 0:3].astype(np.float64).copy()
    quat = obs[0, 9:13].astype(np.float64)
    r_wb = quat_to_rot(quat)
    vel  = r_wb @ obs[0, 3:6].astype(np.float64)
    acc  = r_wb @ obs[0, 6:9].astype(np.float64)
    return pos, vel, acc


def yaw_frame_from_start_to_goal(start, goal, eps=1e-6):
    dx, dy = float(goal[0]-start[0]), float(goal[1]-start[1])
    if dx*dx + dy*dy < eps*eps:
        return None, None
    yaw = float(np.arctan2(dy, dx))
    cy, sy = float(np.cos(0.5*yaw)), float(np.sin(0.5*yaw))
    q = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
    return q, quat_to_rot(q)


def build_state_goal(cur, goal_pos_world, get_body_frame_fn,
                     quat_wb_override=None, r_wb_override=None):
    if quat_wb_override is not None:
        quat_wb, r_wb = quat_wb_override.astype(np.float64), r_wb_override.astype(np.float64)
    else:
        quat_wb, r_wb = get_body_frame_fn(cur.vel, cur.acc, cur.jerk)
    r_bw = r_wb.T
    state_pos   = np.zeros(3, dtype=np.float32)
    state_vel   = (r_bw @ cur.vel).astype(np.float32)
    state_acc   = (r_bw @ cur.acc).astype(np.float32)
    state_jerk  = (r_bw @ cur.jerk).astype(np.float32)
    goal_pos_b  = (r_bw @ (goal_pos_world - cur.pos)).astype(np.float32)
    goal_vel_b  = np.zeros(3, dtype=np.float32)
    goal_acc_b  = np.zeros(3, dtype=np.float32)
    state_goal  = np.concatenate([state_pos, state_vel, state_acc, state_jerk,
                                   quat_wb.astype(np.float32),
                                   goal_pos_b, goal_vel_b, goal_acc_b])
    return state_goal, quat_wb, r_wb


# ---------------------------------------------------------------------------
# FlowPilot inference (verbatim from test_flowpilot_env_loop.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_one_step(model, vae_encoder, basis, scheduler,
                   cond_depth, state_goal, sg_mean, sg_std,
                   free_mean, free_std, v_cmd, num_steps, device):
    cond_frame    = torch.from_numpy(cond_depth.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    sg_t          = torch.from_numpy(state_goal.astype(np.float32)).unsqueeze(0).to(device)
    sg_norm       = (sg_t - sg_mean) / sg_std
    init_pva      = sg_t[:, 0:9]
    constrained   = basis.constrained_coeffs(init_pva)
    v_cmd_t       = torch.tensor([v_cmd], dtype=torch.float32, device=device)

    cond_latent   = vae_encoder.encode(cond_frame.unsqueeze(1))
    video_latent  = torch.randn(1, 48, 3, 6, 10, device=device)
    video_latent[:, :, 0:1] = cond_latent
    free_latent   = torch.randn(1, 5, 3, device=device)

    sigmas   = scheduler.inference_sigmas(num_steps).to(device)
    t_linear = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_ids    = (t_linear[i] * 999).long().clamp(0, 999).unsqueeze(0)
        d_sigma  = sigmas[i + 1] - sigmas[i]
        v_pred, c_pred = model(video_latent, sg_norm, v_cmd_t, free_latent, t_ids, t_ids)
        video_latent = video_latent.clone()
        video_latent[:, :, 1:] = video_latent[:, :, 1:] + v_pred * d_sigma
        video_latent[:, :, 0:1] = cond_latent
        free_latent = free_latent + c_pred * d_sigma

    free_denorm  = free_latent * free_std + free_mean
    wp_body      = basis.expand_to_12d(free_denorm, constrained)[0]
    return wp_body.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# PLY loader (numpy only, no open3d dependency)
# ---------------------------------------------------------------------------

def _load_ply_xyz(path: str) -> np.ndarray:
    """Read xyz points from a binary or ASCII PLY file. Returns [N,3] float32."""
    with open(path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        header = '\n'.join(header_lines)

        # Count vertices
        n_verts = 0
        for line in header_lines:
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
                break

        # Detect binary vs ASCII and property order
        is_binary_le = 'format binary_little_endian' in header
        is_binary_be = 'format binary_big_endian' in header

        props = []
        in_vertex = False
        for line in header_lines:
            if line.startswith('element vertex'):
                in_vertex = True
            elif line.startswith('element') and in_vertex:
                break
            elif in_vertex and line.startswith('property'):
                parts = line.split()
                props.append((parts[1], parts[2]))  # (type, name)

        prop_names = [p[1] for p in props]
        prop_types = [p[0] for p in props]
        type_map   = {'float': 'f4', 'float32': 'f4', 'double': 'f8',
                      'int': 'i4', 'uint': 'u4', 'uchar': 'u1', 'short': 'i2'}
        dtype = np.dtype([(name, type_map.get(t, 'f4')) for name, t in zip(prop_names, prop_types)])

        if is_binary_le or is_binary_be:
            raw  = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)
            if is_binary_be:
                raw = raw.byteswap().newbyteorder()
        else:
            rows = [f.readline().decode().split() for _ in range(n_verts)]
            raw  = np.array([tuple(r) for r in rows], dtype=dtype)

    x = raw['x'].astype(np.float32)
    y = raw['y'].astype(np.float32)
    z = raw['z'].astype(np.float32)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='FlowPilot RViz visualization loop.')
    p.add_argument('--flowpilot_root',    type=str, default='/root/workspace/poly-action-rw')
    p.add_argument('--device',            type=str, default='cuda')
    p.add_argument('--phase2_ckpt',       type=str, default='poly-action-rw/ckpt/ckpt_final.pt')
    p.add_argument('--vae_pth',           type=str, default='poly-action-rw/ckpt/Wan2.2_VAE.pth')
    p.add_argument('--action_stats_path', type=str, default='poly-action-rw/ckpt/action_stats.pt')
    p.add_argument('--coeff_stats_path',  type=str, default='poly-action-rw/ckpt/coeff_stats.pt')
    p.add_argument('--num_steps',         type=int, default=5)
    p.add_argument('--rollout_points',    type=int, default=24)
    p.add_argument('--v_cmd',             type=float, default=6.0)
    p.add_argument('--scene_id',          type=int, default=0)
    p.add_argument('--goal_reach_dist',   type=float, default=4.0)
    p.add_argument('--camera_pitch_deg',  type=float, default=0.0)
    p.add_argument('--compile_model',     action='store_true')
    return p.parse_args()


def main():
    faulthandler.enable(all_threads=True)
    global _goal_pos_world, _goal_received

    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    def ckpt(path):
        return path if os.path.isabs(path) else str(script_dir / path)

    # --- ROS init ---
    rospy.init_node('flowpilot_rviz', anonymous=False)
    traj_pub   = rospy.Publisher('/flowpilot/best_traj_visual', PointCloud2, queue_size=1)
    map_pub    = rospy.Publisher('/local_map',                  PointCloud2, queue_size=1, latch=True)
    depth_pub  = rospy.Publisher('/depth_image',                Image,       queue_size=1)
    marker_pub = rospy.Publisher('/uav_mesh',                   Marker,      queue_size=1)
    goal_pub   = rospy.Publisher('/flowpilot/goal_marker',      Marker,      queue_size=1)
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, _callback_set_goal, queue_size=1)
    tf_br = tf2_ros.TransformBroadcaster()

    ros_thread = threading.Thread(target=rospy.spin, daemon=True)
    ros_thread.start()
    print('[RViz] ROS node started. Set goal in RViz with 2D Nav Goal.')

    # --- FlowPilot imports ---
    flowpilot_root = ckpt(args.flowpilot_root)
    if flowpilot_root not in sys.path:
        sys.path.insert(0, flowpilot_root)
    from flowpilot.scheduler import FlowMatchScheduler
    from flowpilot.models.flowpilot import FlowPilotPhase2
    from flowpilot.models.vae_encoder import WanVAEEncoder
    from flowpilot.models.poly_action import BernsteinActionBasis
    from flowpilot.coords import body_to_world, get_body_frame

    # --- Flightmare env ---
    flightmare_path = os.environ.get('FLIGHTMARE_PATH', str(script_dir.parent))
    os.environ.setdefault('FLIGHTMARE_PATH', flightmare_path)

    from flightgym import QuadrotorEnv_v1
    from flightpolicy.envs import vec_env_wrapper as wrapper

    cfg_path = os.path.join(flightmare_path, 'flightlib', 'configs', 'vec_env.yaml')
    cfg = YAML().load(open(cfg_path))
    cfg['env']['num_envs'] = 1
    cfg['env']['num_threads'] = 1
    cfg['env']['render'] = True
    cfg['env']['supervised'] = False
    cfg['env']['imitation'] = False
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env.connectUnity()
    env.setMapID(np.array([args.scene_id]))
    wb = env.world_box.astype(np.float64)
    print(f'[env] scene {args.scene_id} ready, world_box={wb.tolist()}')

    env.spawnTreesAndSavePointcloud(args.scene_id, spacing=3.5)
    env.render()

    # --- Publish scene point cloud (latched, published once) ---
    ply_path = os.path.join(flightmare_path, 'run', 'yopo_sim', f'pointcloud-{args.scene_id}.ply')
    if os.path.exists(ply_path):
        try:
            pts = _load_ply_xyz(ply_path)
            map_header = Header()
            map_header.stamp = rospy.Time.now()
            map_header.frame_id = 'world'
            map_pub.publish(point_cloud2.create_cloud_xyz32(map_header, pts))
            print(f'[map] published {len(pts)} points from {ply_path}')
        except Exception as e:
            print(f'[map] failed to load/publish pointcloud: {e}')
    else:
        print(f'[map] PLY not found: {ply_path} (skip)')

    # --- Model ---
    model = FlowPilotPhase2(
        num_layers=12, d_video=512, d_action=256, num_heads=8,
        ffn_dim_video=2048, ffn_dim_action=1024, z_dim=48,
        num_actions=5, latent_t=3, grid_h=6, grid_w=10,
    ).to(args.device)
    ckpt_d = torch.load(ckpt(args.phase2_ckpt), map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt_d['model'])
    model.eval()
    if args.compile_model:
        model = torch.compile(model, mode='reduce-overhead')
        print('[model] compiled with torch.compile(mode=reduce-overhead)')

    vae_encoder = WanVAEEncoder(ckpt(args.vae_pth), dtype=torch.bfloat16, device=args.device)
    vae_encoder.eval()

    basis     = BernsteinActionBasis(degree=7, T=1.6, num_waypoints=48, hz=30, device=args.device)
    scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    astats   = torch.load(ckpt(args.action_stats_path), map_location=args.device, weights_only=True)
    cstats   = torch.load(ckpt(args.coeff_stats_path),  map_location=args.device, weights_only=True)
    sg_mean  = astats['sg_mean'].to(args.device)
    sg_std   = astats['sg_std'].to(args.device)
    free_mean = cstats['free_mean'].to(args.device)
    free_std  = cstats['free_std'].to(args.device)
    print('[model] all weights loaded.')

    # Warm-up
    torch.set_grad_enabled(False)
    _dummy_depth = np.zeros((96, 160), dtype=np.float32)
    _dummy_sg    = np.zeros(25, dtype=np.float32)
    infer_one_step(model, vae_encoder, basis, scheduler,
                   _dummy_depth, _dummy_sg, sg_mean, sg_std,
                   free_mean, free_std, args.v_cmd, args.num_steps, args.device)
    print('[model] warm-up done. Waiting for goal from RViz...')

    # Camera pitch
    pitch_rad = float(args.camera_pitch_deg) * math.pi / 180.0
    q_pitch   = np.array([math.cos(0.5*pitch_rad), 0.0, math.sin(0.5*pitch_rad), 0.0], dtype=np.float64)

    # --- Main loop ---
    while not rospy.is_shutdown():
        # Wait for goal
        with _goal_lock:
            goal_ready = _goal_received
            goal_pos   = _goal_pos_world.copy() if _goal_received else None
        if not goal_ready:
            time.sleep(0.1)
            continue

        # Reset env, sample start from env.reset()
        obs  = env.reset()
        pos0, vel0, acc0 = obs_to_world_pva(obs)
        cur  = WorldState(pos=pos0, vel=np.zeros(3), acc=np.zeros(3), jerk=np.zeros(3))

        # Initial yaw pointing toward goal
        init_q, init_r = yaw_frame_from_start_to_goal(cur.pos, goal_pos)
        if init_q is not None:
            state_goal, quat_wb, r_wb = build_state_goal(
                cur, goal_pos, get_body_frame,
                quat_wb_override=init_q, r_wb_override=init_r)
        else:
            state_goal, quat_wb, r_wb = build_state_goal(cur, goal_pos, get_body_frame)

        quat_render = _safe_quat(quat_wb)
        if abs(pitch_rad) > 1e-12:
            quat_render = _safe_quat(quat_mul(quat_render, q_pitch))
        env.setState(cur.pos, cur.vel, cur.acc, quat_render)
        env.render()
        cond_depth = env.getDepthImage()[0][0].astype(np.float32)

        print(f'[run] start=({cur.pos[0]:.1f},{cur.pos[1]:.1f},{cur.pos[2]:.1f}) '
              f'goal=({goal_pos[0]:.1f},{goal_pos[1]:.1f},{goal_pos[2]:.1f})')

        loop_idx = 0
        while not rospy.is_shutdown():
            # Check for new goal mid-trajectory
            with _goal_lock:
                if _goal_pos_world is not goal_pos:
                    goal_pos   = _goal_pos_world.copy()
                    print(f'[run] goal updated mid-traj: ({goal_pos[0]:.1f},{goal_pos[1]:.1f})')

            # Inference
            t0 = time.perf_counter()
            wp_body = infer_one_step(
                model, vae_encoder, basis, scheduler,
                cond_depth, state_goal, sg_mean, sg_std,
                free_mean, free_std, args.v_cmd, args.num_steps, args.device)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            if not np.all(np.isfinite(wp_body)):
                print(f'[run loop {loop_idx}] NaN/Inf in wp_body, resetting.')
                break

            wp_world = body_to_world(wp_body, cur.pos, r_wb).astype(np.float64)

            # Clamp to world box
            wp_world[:, 0] = np.clip(wp_world[:, 0], wb[0]+0.2, wb[3]-0.2)
            wp_world[:, 1] = np.clip(wp_world[:, 1], wb[1]+0.2, wb[4]-0.2)
            wp_world[:, 2] = np.clip(wp_world[:, 2], max(0.05, wb[2]+0.05), wb[5]-0.1)

            # Publish trajectory to RViz
            publish_trajectory(traj_pub, wp_world)
            publish_goal_marker(goal_pub, goal_pos)

            print(f'[run loop {loop_idx}] infer={infer_ms:.1f}ms '
                  f'dist={np.linalg.norm(goal_pos - cur.pos):.2f}m')

            # Rollout in Flightmare
            exec_n = min(args.rollout_points, wp_world.shape[0])
            for k in range(exec_n):
                pk = wp_world[k, 0:3]
                vk = wp_world[k, 3:6]
                ak = wp_world[k, 6:9]
                jk = wp_world[k, 9:12]
                if not (np.all(np.isfinite(pk)) and np.all(np.isfinite(vk)) and np.all(np.isfinite(ak))):
                    break
                quat_k, _ = get_body_frame(vk, ak, jk)
                quat_k = _safe_quat(np.asarray(quat_k, dtype=np.float64))
                if abs(pitch_rad) > 1e-12:
                    quat_k = _safe_quat(quat_mul(quat_k, q_pitch))
                env.setState(pk, vk, ak, quat_k)
                env.render()
                cond_depth = env.getDepthImage()[0][0].astype(np.float32)

                # Publish depth + drone pose to RViz at each render step
                publish_depth(depth_pub, cond_depth)
                publish_drone_marker(marker_pub, pk, quat_k)
                broadcast_tf(tf_br, pk, quat_k)

            # Advance state to last executed point
            last = wp_world[exec_n - 1]
            cur  = WorldState(pos=last[0:3].copy(), vel=last[3:6].copy(),
                               acc=last[6:9].copy(), jerk=last[9:12].copy())
            state_goal, quat_wb, r_wb = build_state_goal(cur, goal_pos, get_body_frame)

            dist = float(np.linalg.norm(goal_pos - cur.pos))
            if dist < args.goal_reach_dist:
                print(f'[run] arrived! dist={dist:.2f}m. Waiting for next goal...')
                with _goal_lock:
                    _goal_received = False
                    _goal_pos_world = None
                break

            loop_idx += 1


if __name__ == '__main__':
    main()
