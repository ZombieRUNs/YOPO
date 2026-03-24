"""
Minimal closed-loop test env for FlowPilot poly-action in Flightmare.

Loop:
  1) Build model input from current state + current depth frame.
  2) Run FlowPilot Phase2 inference once (predict 48 body-frame waypoints at 30 Hz).
  3) Convert 48 points to world frame and render them in env via setState + render.
  4) Use the last sampled point (pos/vel/acc/jerk) as next-step state input.

This script is intentionally lightweight for quick integration validation.
"""

import argparse
import faulthandler
import os
import sys
from datetime import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import cv2
from ruamel.yaml import YAML, RoundTripDumper, dump


@dataclass
class WorldState:
    pos: np.ndarray   # [3]
    vel: np.ndarray   # [3]
    acc: np.ndarray   # [3]
    jerk: np.ndarray  # [3]


def _safe_quat(quat: np.ndarray) -> np.ndarray:
    """Return a valid quaternion; fallback to identity if invalid."""
    if quat.shape != (4,) or (not np.all(np.isfinite(quat))):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = float(np.linalg.norm(quat))
    if n < 1e-9 or (not np.isfinite(n)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (quat / n).astype(np.float64)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] -> 3x3 rotation matrix (world <- body)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply in wxyz convention: q = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def obs_to_world_pva(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Flightmare obs [1,13] into world-frame p/v/a."""
    pos = obs[0, 0:3].astype(np.float64).copy()
    quat = obs[0, 9:13].astype(np.float64)
    r_wb = quat_to_rot(quat)
    vel = r_wb @ obs[0, 3:6].astype(np.float64)
    acc = r_wb @ obs[0, 6:9].astype(np.float64)
    return pos, vel, acc


def import_flowpilot_modules(flowpilot_root: str):
    sys.path.insert(0, flowpilot_root)
    from flowpilot.scheduler import FlowMatchScheduler
    from flowpilot.models.flowpilot import FlowPilotPhase2
    from flowpilot.models.vae_encoder import WanVAEEncoder
    from flowpilot.models.poly_action import BernsteinActionBasis
    from flowpilot.coords import body_to_world, get_body_frame
    return FlowMatchScheduler, FlowPilotPhase2, WanVAEEncoder, BernsteinActionBasis, body_to_world, get_body_frame


@torch.no_grad()
def infer_one_step(
    model,
    vae_encoder,
    basis,
    scheduler,
    cond_depth: np.ndarray,        # [96,160], [0,1]
    state_goal: np.ndarray,        # [25]
    sg_mean: torch.Tensor,         # [25]
    sg_std: torch.Tensor,          # [25]
    free_mean: torch.Tensor,       # [5,3]
    free_std: torch.Tensor,        # [5,3]
    v_cmd: float,
    num_steps: int,
    device: str,
) -> np.ndarray:
    """Return predicted body-frame waypoints [48,12]."""
    cond_frame = torch.from_numpy(cond_depth.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    state_goal_t = torch.from_numpy(state_goal.astype(np.float32)).unsqueeze(0).to(device)

    sg_norm = (state_goal_t - sg_mean) / sg_std
    init_pva = state_goal_t[:, 0:9]
    constrained = basis.constrained_coeffs(init_pva)
    v_cmd_t = torch.tensor([v_cmd], dtype=torch.float32, device=device)

    cond_latent = vae_encoder.encode(cond_frame.unsqueeze(1))  # [1,48,1,6,10]
    video_latent = torch.randn(1, 48, 3, 6, 10, device=device)
    video_latent[:, :, 0:1] = cond_latent
    free_latent = torch.randn(1, 5, 3, device=device)

    sigmas = scheduler.inference_sigmas(num_steps).to(device)
    t_linear = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_ids = (t_linear[i] * 999).long().clamp(0, 999).unsqueeze(0)
        d_sigma = sigmas[i + 1] - sigmas[i]
        v_pred, c_pred = model(video_latent, sg_norm, v_cmd_t, free_latent, t_ids, t_ids)
        video_latent = video_latent.clone()
        video_latent[:, :, 1:] = video_latent[:, :, 1:] + v_pred * d_sigma
        video_latent[:, :, 0:1] = cond_latent
        free_latent = free_latent + c_pred * d_sigma

    free_denorm = free_latent * free_std + free_mean
    waypoints_body = basis.expand_to_12d(free_denorm, constrained)[0]  # [48,12]
    return waypoints_body.cpu().numpy().astype(np.float32)


def build_state_goal(
    cur: WorldState,
    goal_pos_world: np.ndarray,
    get_body_frame_fn,
    quat_wb_override: np.ndarray = None,
    r_wb_override: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build FlowPilot state_goal[25] from current world state and fixed goal."""
    if quat_wb_override is not None and r_wb_override is not None:
        quat_wb = quat_wb_override.astype(np.float64)
        r_wb = r_wb_override.astype(np.float64)
    else:
        quat_wb, r_wb = get_body_frame_fn(cur.vel, cur.acc, cur.jerk)  # world <- body
    r_bw = r_wb.T

    state_pos = np.zeros(3, dtype=np.float32)
    state_vel = (r_bw @ cur.vel).astype(np.float32)
    state_acc = (r_bw @ cur.acc).astype(np.float32)
    state_jerk = (r_bw @ cur.jerk).astype(np.float32)
    state_12d = np.concatenate([state_pos, state_vel, state_acc, state_jerk], axis=0)

    goal_pos_body = (r_bw @ (goal_pos_world - cur.pos)).astype(np.float32)
    goal_vel_body = np.zeros(3, dtype=np.float32)
    goal_acc_body = np.zeros(3, dtype=np.float32)
    goal_9d = np.concatenate([goal_pos_body, goal_vel_body, goal_acc_body], axis=0)

    state_goal = np.concatenate([state_12d, quat_wb.astype(np.float32), goal_9d], axis=0)
    return state_goal, quat_wb, r_wb


def yaw_frame_from_start_to_goal(
    start_pos_world: np.ndarray,
    goal_pos_world: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return yaw-only body frame (quat, R) whose x-axis points from start to goal on XY plane."""
    dx = float(goal_pos_world[0] - start_pos_world[0])
    dy = float(goal_pos_world[1] - start_pos_world[1])
    if (dx * dx + dy * dy) < eps * eps:
        return None, None
    yaw = float(np.arctan2(dy, dx))
    cy = float(np.cos(0.5 * yaw))
    sy = float(np.sin(0.5 * yaw))
    quat_wb = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)  # yaw around +Z
    r_wb = quat_to_rot(quat_wb)
    return quat_wb, r_wb


def sample_start_goal_world(
    env,
    planner,
    min_goal_dist: float,
    max_goal_dist: float,
    max_tries: int = 300,
) -> Tuple[WorldState, np.ndarray]:
    """Sample collision-free start/goal using Flightmare reset + planner.isFree."""
    for _ in range(max_tries):
        obs_s = env.reset()
        sp, sv, sa = obs_to_world_pva(obs_s)
        obs_g = env.reset()
        gp, _, _ = obs_to_world_pva(obs_g)
        dist = np.linalg.norm(gp - sp)
        if not (min_goal_dist <= dist <= max_goal_dist):
            continue
        if (not planner.isFree(sp)) or (not planner.isFree(gp)):
            continue
        cur = WorldState(pos=sp, vel=sv, acc=sa, jerk=np.zeros(3, dtype=np.float64))
        return cur, gp
    return None, None


def save_depth_video(depth_frames: List[np.ndarray], out_path: Path, fps: float) -> None:
    """Save depth frames [M,H,W] uint8 to mp4."""
    if len(depth_frames) == 0:
        return
    h, w = depth_frames[0].shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    for d in depth_frames:
        if d.dtype != np.uint8:
            d = np.clip(d, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
    writer.release()


def main():
    faulthandler.enable(all_threads=True)

    parser = argparse.ArgumentParser(description="FlowPilot poly-action closed-loop env test.")
    parser.add_argument("--flowpilot_root", type=str, default="/root/workspace/poly-action-rw")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--phase2_ckpt", type=str, default="/root/workspace/YOPO/run/poly-action-rw/ckpt/ckpt_final.pt")
    parser.add_argument("--vae_pth", type=str, default="/root/workspace/YOPO/run/poly-action-rw/ckpt/Wan2.2_VAE.pth")
    parser.add_argument("--action_stats_path", type=str, default="/root/workspace/YOPO/run/poly-action-rw/ckpt/action_stats.pt")
    parser.add_argument("--coeff_stats_path", type=str, default="/root/workspace/YOPO/run/poly-action-rw/ckpt/coeff_stats.pt")
    parser.add_argument("--num_loops", type=int, default=100)
    parser.add_argument("--num_trajs", type=int, default=10, help="How many independent trajectories to run in this scene.")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--rollout_points", type=int, default=24, help="How many leading predicted points to execute each loop (1..48).")
    parser.add_argument("--v_cmd", type=float, default=4.0)
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--spawn_trees_before_reset", dest="spawn_trees_before_reset", action="store_true")
    parser.add_argument("--no_spawn_trees_before_reset", dest="spawn_trees_before_reset", action="store_false")
    parser.add_argument("--min_goal_dist", type=float, default=30.0)
    parser.add_argument("--max_goal_dist", type=float, default=40.0)
    parser.add_argument("--goal_reach_dist", type=float, default=2.0, help="If straight-line distance to goal is below this (m), end current traj early and reset to next traj.")
    parser.add_argument("--pair_max_tries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_npz", type=str, default="")
    parser.add_argument("--save_depth_mp4", dest="save_depth_mp4", action="store_true", help="Save per-traj rendered depth frames to mp4.")
    parser.add_argument("--no_save_depth_mp4", dest="save_depth_mp4", action="store_false")
    parser.add_argument("--depth_video_fps", type=float, default=30.0, help="FPS for saved depth mp4.")
    parser.add_argument("--launch_unity", action="store_true")
    parser.add_argument("--show_depth", action="store_true")
    parser.add_argument("--depth_window_w", type=int, default=480, help="cv2 depth window width when --show_depth is enabled.")
    parser.add_argument("--depth_window_h", type=int, default=300, help="cv2 depth window height when --show_depth is enabled.")
    parser.add_argument("--camera_pitch_deg", type=float, default=0.0, help="Render-only camera pitch offset (deg). Applied in env.setState only.")
    parser.add_argument("--compile_model", action="store_true", help="Enable torch.compile for FlowPilotPhase2 inference.")
    parser.add_argument(
        "--debug_stop_after",
        type=str,
        default="",
        choices=["", "imports", "env", "model", "vae", "stats", "init", "infer"],
        help="Stop after a debug stage to isolate segfault location.",
    )
    parser.set_defaults(spawn_trees_before_reset=True)
    parser.set_defaults(save_depth_mp4=True)
    args = parser.parse_args()
    if args.rollout_points < 1 or args.rollout_points > 48:
        raise ValueError("--rollout_points must be in [1, 48].")
    if args.depth_window_w <= 0 or args.depth_window_h <= 0:
        raise ValueError("--depth_window_w and --depth_window_h must be positive.")
    if args.num_trajs < 1:
        raise ValueError("--num_trajs must be >= 1.")
    if args.depth_video_fps <= 0:
        raise ValueError("--depth_video_fps must be positive.")
    if not np.isfinite(args.camera_pitch_deg):
        raise ValueError("--camera_pitch_deg must be finite.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.show_depth:
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("depth", args.depth_window_w, args.depth_window_h)
    pitch_rad = float(args.camera_pitch_deg) * np.pi / 180.0
    q_pitch = np.array([np.cos(0.5 * pitch_rad), 0.0, np.sin(0.5 * pitch_rad), 0.0], dtype=np.float64)

    flightmare_path = os.environ.get("FLIGHTMARE_PATH", str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("FLIGHTMARE_PATH", flightmare_path)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).resolve().parent
    run_out_dir = script_dir / "flowpilot_env_loop" / run_tag
    if args.save_npz:
        user_base = Path(args.save_npz)
        base_stem = user_base.stem if user_base.stem else "flowpilot_env_loop_debug"
        base_suffix = user_base.suffix if user_base.suffix else ".npz"
    else:
        base_stem = "flowpilot_env_loop_debug"
        base_suffix = ".npz"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dbg] run output dir: {run_out_dir}", flush=True)

    FlowMatchScheduler, FlowPilotPhase2, WanVAEEncoder, BernsteinActionBasis, body_to_world_fn, get_body_frame_fn = (
        import_flowpilot_modules(args.flowpilot_root)
    )
    print("[dbg] imports ok", flush=True)
    if args.debug_stop_after == "imports":
        return

    from flightgym import QuadrotorEnv_v1
    from flightpolicy.envs import vec_env_wrapper as wrapper

    if args.launch_unity:
        os.system(os.path.join(flightmare_path, "flightrender/RPG_Flightmare/flightmare.x86_64 &"))

    cfg_path = os.path.join(flightmare_path, "flightlib", "configs", "vec_env.yaml")
    cfg = YAML().load(open(cfg_path, "r"))
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = True
    cfg["env"]["supervised"] = False
    cfg["env"]["imitation"] = False
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env.connectUnity()
    env.setMapID(np.array([args.scene_id]))
    wb = env.world_box.astype(np.float64)  # [xmin,ymin,zmin,xmax,ymax,zmax]
    print(f"[dbg] env ok world_box={wb.tolist()}", flush=True)
    if args.debug_stop_after == "env":
        return

    if args.spawn_trees_before_reset:
        print("[dbg] spawning trees before first reset...", flush=True)
        env.spawnTreesAndSavePointcloud(args.scene_id, spacing=3.16)
        env.render()
        print("[dbg] spawn trees done", flush=True)

    # Build GCOPTER occupancy checker (same pattern as collect_gcopter_trajs.py).
    import gcopter_planner
    planner_cfg = os.path.join(flightmare_path, "flightlib", "configs", "gcopter_config.yaml")
    planner = gcopter_planner.GcopterPlanner(planner_cfg)
    map_bound = [
        float(wb[0]), float(wb[3]),   # xmin, xmax
        float(wb[1]), float(wb[4]),   # ymin, ymax
        max(0.0, float(wb[2])),       # zmin
        min(8.0, float(wb[5])),       # zmax
    ]
    ply_path = os.path.join(flightmare_path, "run", "yopo_sim", f"pointcloud-{args.scene_id}.ply")
    planner.loadScene(ply_path, map_bound, 0.25, 0.5)
    print(f"[dbg] planner scene loaded: {ply_path}", flush=True)

    # Load FlowPilot assets.
    model = FlowPilotPhase2(
        num_layers=12, d_video=512, d_action=256, num_heads=8,
        ffn_dim_video=2048, ffn_dim_action=1024, z_dim=48,
        num_actions=5, latent_t=3, grid_h=6, grid_w=10,
    ).to(args.device)
    ckpt = torch.load(args.phase2_ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.compile_model:
        model = torch.compile(model, mode="reduce-overhead")
        print("[dbg] model compiled with torch.compile(mode=reduce-overhead)", flush=True)
    print("[dbg] model ok", flush=True)
    if args.debug_stop_after == "model":
        return

    vae_encoder = WanVAEEncoder(args.vae_pth, dtype=torch.bfloat16, device=args.device)
    vae_encoder.eval()
    print("[dbg] vae ok", flush=True)
    if args.debug_stop_after == "vae":
        return

    basis = BernsteinActionBasis(degree=7, T=1.6, num_waypoints=48, hz=30, device=args.device)
    scheduler = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    astats = torch.load(args.action_stats_path, map_location=args.device, weights_only=True)
    cstats = torch.load(args.coeff_stats_path, map_location=args.device, weights_only=True)
    sg_mean = astats["sg_mean"].to(args.device)
    sg_std = astats["sg_std"].to(args.device)
    free_mean = cstats["free_mean"].to(args.device)
    free_std = cstats["free_std"].to(args.device)
    print("[dbg] stats ok", flush=True)
    if args.debug_stop_after == "stats":
        return

    for traj_idx in range(args.num_trajs):
        cur, goal_pos_world = sample_start_goal_world(
            env=env,
            planner=planner,
            min_goal_dist=args.min_goal_dist,
            max_goal_dist=args.max_goal_dist,
            max_tries=args.pair_max_tries,
        )
        if cur is None:
            print(f"[traj {traj_idx:03d}] failed to sample collision-free start/goal, skip.", flush=True)
            continue

        # Render initial condition frame. Force initial yaw to point from start to goal.
        init_quat_wb, init_r_wb = yaw_frame_from_start_to_goal(cur.pos, goal_pos_world)
        if init_quat_wb is not None and init_r_wb is not None:
            state_goal, quat_wb, r_wb = build_state_goal(
                cur,
                goal_pos_world,
                get_body_frame_fn,
                quat_wb_override=init_quat_wb,
                r_wb_override=init_r_wb,
            )
        else:
            state_goal, quat_wb, r_wb = build_state_goal(cur, goal_pos_world, get_body_frame_fn)
        quat_render = _safe_quat(np.asarray(quat_wb, dtype=np.float64))
        if abs(pitch_rad) > 1e-12:
            quat_render = _safe_quat(quat_mul(quat_render, q_pitch))
        env.setState(cur.pos, cur.vel, cur.acc, quat_render)
        env.render()
        cond_depth = env.getDepthImage()[0][0].astype(np.float32)  # [96,160]
        depth_frames_u8: List[np.ndarray] = [np.clip(cond_depth * 255.0, 0, 255).astype(np.uint8)]
        if args.show_depth:
            depth_vis = depth_frames_u8[-1]
            cv2.imshow("depth", depth_vis)
            cv2.waitKey(1)
        print(f"[traj {traj_idx:03d}] init state/render ok", flush=True)
        if args.debug_stop_after == "init":
            if args.show_depth:
                cv2.destroyAllWindows()
            return

        traj_world_steps: List[np.ndarray] = []
        last_states: List[np.ndarray] = []

        for loop_idx in range(args.num_loops):
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            wp_body = infer_one_step(
                model=model,
                vae_encoder=vae_encoder,
                basis=basis,
                scheduler=scheduler,
                cond_depth=cond_depth,
                state_goal=state_goal,
                sg_mean=sg_mean,
                sg_std=sg_std,
                free_mean=free_mean,
                free_std=free_std,
                v_cmd=args.v_cmd,
                num_steps=args.num_steps,
                device=args.device,
            )  # [48,12]
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[dbg] infer ok traj={traj_idx} loop={loop_idx}", flush=True)
            print(f"[perf] traj={traj_idx} loop={loop_idx} infer_ms={infer_ms:.2f}", flush=True)
            if args.debug_stop_after == "infer":
                return
            if not np.all(np.isfinite(wp_body)):
                print(f"[traj {traj_idx:03d} loop {loop_idx:03d}] invalid wp_body (NaN/Inf), stop.", flush=True)
                break

            wp_world = body_to_world_fn(wp_body, cur.pos, r_wb).astype(np.float64)  # [48,12]
            if not np.all(np.isfinite(wp_world)):
                print(f"[traj {traj_idx:03d} loop {loop_idx:03d}] invalid wp_world (NaN/Inf), stop.", flush=True)
                break

            # Safety clamps to avoid sending extreme states into C++ env.
            wp_world[:, 0] = np.clip(wp_world[:, 0], wb[0] + 0.2, wb[3] - 0.2)
            wp_world[:, 1] = np.clip(wp_world[:, 1], wb[1] + 0.2, wb[4] - 0.2)
            wp_world[:, 2] = np.clip(wp_world[:, 2], max(0.05, wb[2] + 0.05), wb[5] - 0.1)
            wp_world[:, 3:6] = np.clip(wp_world[:, 3:6], -8.0, 8.0)
            wp_world[:, 6:9] = np.clip(wp_world[:, 6:9], -20.0, 20.0)
            wp_world[:, 9:12] = np.clip(wp_world[:, 9:12], -40.0, 40.0)
            traj_world_steps.append(wp_world.astype(np.float32))

            # Rollout in env: execute only leading rollout_points waypoints.
            exec_n = min(args.rollout_points, wp_world.shape[0])
            for k in range(exec_n):
                pk = wp_world[k, 0:3]
                vk = wp_world[k, 3:6]
                ak = wp_world[k, 6:9]
                jk = wp_world[k, 9:12]
                quat_k, _ = get_body_frame_fn(vk, ak, jk)
                quat_k = _safe_quat(np.asarray(quat_k, dtype=np.float64))
                if abs(pitch_rad) > 1e-12:
                    quat_k = _safe_quat(quat_mul(quat_k, q_pitch))
                if not (np.all(np.isfinite(pk)) and np.all(np.isfinite(vk)) and np.all(np.isfinite(ak))):
                    print(f"[traj {traj_idx:03d} loop {loop_idx:03d}] invalid state at k={k}, stop.", flush=True)
                    break
                env.setState(pk, vk, ak, quat_k)
                env.render()
                cond_depth = env.getDepthImage()[0][0].astype(np.float32)
                depth_frames_u8.append(np.clip(cond_depth * 255.0, 0, 255).astype(np.uint8))
                if args.show_depth:
                    depth_vis = depth_frames_u8[-1]
                    cv2.imshow("depth", depth_vis)
                    cv2.waitKey(1)

            # Next-step state from the last executed point.
            last = wp_world[exec_n - 1]
            cur = WorldState(
                pos=last[0:3].copy(),
                vel=last[3:6].copy(),
                acc=last[6:9].copy(),
                jerk=last[9:12].copy(),
            )
            last_states.append(np.concatenate([cur.pos, cur.vel, cur.acc, cur.jerk], axis=0).astype(np.float32))
            state_goal, quat_wb, r_wb = build_state_goal(cur, goal_pos_world, get_body_frame_fn)

            dist_to_goal = float(np.linalg.norm(goal_pos_world - cur.pos))
            print(
                f"[traj {traj_idx:03d} loop {loop_idx:03d}] "
                f"pos=({cur.pos[0]:.2f},{cur.pos[1]:.2f},{cur.pos[2]:.2f}) "
                f"|v|={np.linalg.norm(cur.vel):.2f} "
                f"exec_pts={exec_n} "
                f"goal_dist={dist_to_goal:.2f}",
                flush=True,
            )
            if dist_to_goal < args.goal_reach_dist:
                print(
                    f"[traj {traj_idx:03d}] reached goal threshold "
                    f"({dist_to_goal:.2f}m < {args.goal_reach_dist:.2f}m), reset to next traj.",
                    flush=True,
                )
                break

        if args.save_npz:
            if args.num_trajs == 1:
                out_path = run_out_dir / f"{base_stem}{base_suffix}"
            else:
                out_path = run_out_dir / f"{base_stem}_traj{traj_idx:03d}{base_suffix}"
            traj_arr = np.stack(traj_world_steps, axis=0) if len(traj_world_steps) > 0 else np.zeros((0, 48, 12), dtype=np.float32)
            last_arr = np.stack(last_states, axis=0) if len(last_states) > 0 else np.zeros((0, 12), dtype=np.float32)
            np.savez_compressed(
                out_path,
                goal_pos_world=goal_pos_world.astype(np.float32),
                traj_world=traj_arr,
                last_states=last_arr,
            )
            print(f"[traj {traj_idx:03d}] saved debug rollout to {out_path}", flush=True)
            if args.save_depth_mp4:
                depth_path_base = out_path.with_suffix("")
                depth_path = depth_path_base.parent / f"{depth_path_base.name}_depth.mp4"
                save_depth_video(depth_frames_u8, depth_path, args.depth_video_fps)
                print(f"[traj {traj_idx:03d}] saved depth video to {depth_path}", flush=True)
        elif args.save_depth_mp4:
            depth_path = run_out_dir / f"{base_stem}_traj{traj_idx:03d}_depth.mp4"
            save_depth_video(depth_frames_u8, depth_path, args.depth_video_fps)
            print(f"[traj {traj_idx:03d}] saved depth video to {depth_path}", flush=True)

    if args.show_depth:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
