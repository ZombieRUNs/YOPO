#!/usr/bin/env python3
"""
Replay one collected GCOPTER trajectory sample:
1) Publish scene point cloud and trajectory to RViz (ROS topics).
2) Export depths.npy / rgb.mp4 to mp4 videos via OpenCV.

Usage example:
  python replay_gcopter_traj.py \
      --dataset_root dataset/gcopter_trajs \
      --scene_id 0 \
      --traj_id 3 \
      --img_width 160 --img_height 96 \
      --video_out /tmp/scene000_traj0003_depth.mp4
"""

import argparse
import os
import struct
import sys
from typing import List, Tuple

import cv2
import numpy as np
from ruamel.yaml import YAML

DEFAULT_IMG_WIDTH = 160
DEFAULT_IMG_HEIGHT = 96
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "replay_gcopter_traj.yaml"
)


PLY_SCALAR_FMT = {
    "char": "b",
    "uchar": "B",
    "int8": "b",
    "uint8": "B",
    "short": "h",
    "ushort": "H",
    "int16": "h",
    "uint16": "H",
    "int": "i",
    "uint": "I",
    "int32": "i",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def load_ply_xyz(ply_path: str) -> np.ndarray:
    """Load XYZ vertices from ASCII or binary_little_endian PLY."""
    with open(ply_path, "rb") as f:
        format_name = None
        vertex_count = None
        properties: List[Tuple[str, str]] = []
        in_vertex = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY (missing end_header): {ply_path}")
            s = line.decode("ascii", errors="ignore").strip()
            if s.startswith("format "):
                format_name = s.split()[1]
            elif s.startswith("element "):
                toks = s.split()
                in_vertex = (len(toks) >= 3 and toks[1] == "vertex")
                if in_vertex:
                    vertex_count = int(toks[2])
                    properties = []
            elif in_vertex and s.startswith("property "):
                toks = s.split()
                if len(toks) == 3:
                    # property <type> <name>
                    properties.append((toks[1], toks[2]))
            elif s == "end_header":
                break

        if format_name not in ("ascii", "binary_little_endian"):
            raise ValueError(f"Unsupported PLY format: {format_name}")
        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"No vertex element in PLY: {ply_path}")

        prop_names = [n for _, n in properties]
        if not all(k in prop_names for k in ("x", "y", "z")):
            raise ValueError(f"PLY vertex has no x/y/z fields: {ply_path}")
        ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")

        if format_name == "ascii":
            pts = np.zeros((vertex_count, 3), dtype=np.float32)
            for i in range(vertex_count):
                parts = f.readline().decode("ascii", errors="ignore").strip().split()
                if len(parts) < len(properties):
                    raise ValueError(f"Bad ASCII PLY vertex line #{i} in {ply_path}")
                pts[i, 0] = float(parts[ix])
                pts[i, 1] = float(parts[iy])
                pts[i, 2] = float(parts[iz])
            return pts

        # binary_little_endian
        if any(ptype not in PLY_SCALAR_FMT for ptype, _ in properties):
            unknown = [ptype for ptype, _ in properties if ptype not in PLY_SCALAR_FMT]
            raise ValueError(f"Unsupported binary PLY property types: {unknown}")
        row_fmt = "<" + "".join(PLY_SCALAR_FMT[ptype] for ptype, _ in properties)
        row_size = struct.calcsize(row_fmt)
        unpack_row = struct.Struct(row_fmt).unpack_from

        pts = np.zeros((vertex_count, 3), dtype=np.float32)
        buf = f.read(vertex_count * row_size)
        if len(buf) < vertex_count * row_size:
            raise ValueError(f"Truncated binary PLY data: {ply_path}")
        off = 0
        for i in range(vertex_count):
            row = unpack_row(buf, off)
            off += row_size
            pts[i, 0] = float(row[ix])
            pts[i, 1] = float(row[iy])
            pts[i, 2] = float(row[iz])
        return pts


def build_sample_paths(dataset_root: str, scene_id: int, traj_id: int):
    scene_dir = os.path.join(dataset_root, f"scene_{scene_id:03d}")
    traj_dir = os.path.join(scene_dir, f"traj_{traj_id:04d}")
    return {
        "scene_dir": scene_dir,
        "traj_dir": traj_dir,
        "ply": os.path.join(scene_dir, "pointcloud.ply"),
        "waypoints": os.path.join(traj_dir, "waypoints.npy"),
        "depths": os.path.join(traj_dir, "depths.npy"),
        "rgb_video": os.path.join(traj_dir, "rgb.mp4"),
    }


def load_rgb_video(video_path: str) -> np.ndarray:
    """Load RGB frames from video file as [N,H,W,3] uint8."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RGB video: {video_path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from RGB video: {video_path}")
    return np.stack(frames, axis=0)


def export_depth_video(
    depths: np.ndarray, out_path: str, fps: float, out_size: Tuple[int, int] = None
) -> None:
    """Save normalized depth [0,1] to a colormapped mp4 video."""
    if depths.ndim != 3:
        raise ValueError(f"depths must be [N,H,W], got {depths.shape}")
    n, h, w = depths.shape
    if out_size is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = out_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    for i in range(n):
        d = np.clip(depths[i].astype(np.float32), 0.0, 1.0)
        # Near = warm color, far = cool color
        gray = ((1.0 - d) * 255.0).astype(np.uint8)
        frame = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        if (out_w, out_h) != (w, h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)
    writer.release()


def export_rgb_video(
    rgbs: np.ndarray, out_path: str, fps: float, out_size: Tuple[int, int] = None
) -> None:
    """Save RGB frames [N,H,W,3] uint8 to an mp4 video."""
    if rgbs.ndim != 4 or rgbs.shape[-1] != 3:
        raise ValueError(f"rgbs must be [N,H,W,3], got {rgbs.shape}")
    n, h, w, _ = rgbs.shape
    if out_size is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = out_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    # OpenCV writer expects BGR order.
    for i in range(n):
        frame_rgb = rgbs[i]
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if (out_w, out_h) != (w, h):
            frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame_bgr)
    writer.release()


def resolve_video_size(img_width: int, img_height: int) -> Tuple[int, int]:
    """Resolve output size (w, h) from CLI args."""
    if img_width <= 0 or img_height <= 0:
        raise ValueError("--img_width and --img_height must be positive integers.")
    return int(img_width), int(img_height)


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
    # roslaunch injects remap args like "__name:=..." and "__log:=...".
    argv = [a for a in sys.argv[1:] if not a.startswith("__")]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Replay one GCOPTER sample in RViz and export depth/RGB videos"
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_root", type=str, default="dataset/gcopter_trajs")
    parser.add_argument("--scene_id", type=int, default=None)
    parser.add_argument("--traj_id", type=int, default=None)
    parser.add_argument("--frame_id", type=str, default="world")
    parser.add_argument("--topic_map", type=str, default="/collect/map")
    parser.add_argument("--topic_path", type=str, default="/collect/path")
    parser.add_argument("--topic_marker", type=str, default="/collect/drone_marker")
    parser.add_argument("--rate", type=float, default=30.0, help="Replay rate in Hz")
    parser.add_argument(
        "--check_collision",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Check trajectory collision against point cloud.",
    )
    parser.add_argument(
        "--collision_radius",
        type=float,
        default=0.5,
        help="Collision radius (meters) for pointcloud-distance check.",
    )
    parser.add_argument(
        "--video_out",
        type=str,
        nargs="?",
        const="",
        default="",
        help="If set, export depths.npy to this mp4 path",
    )
    parser.add_argument(
        "--rgb_video_out",
        type=str,
        nargs="?",
        const="",
        default="",
        help="If set, export RGB video to this mp4 path",
    )
    parser.add_argument(
        "--video_only",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Only export depth/RGB video; skip ROS/RViz publishing.",
    )
    parser.add_argument(
        "--loop",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Loop replay until ROS shutdown.",
    )
    parser.add_argument("--img_width", type=int, default=DEFAULT_IMG_WIDTH)
    parser.add_argument("--img_height", type=int, default=DEFAULT_IMG_HEIGHT)

    valid_keys = {a.dest for a in parser._actions}
    unknown = sorted([k for k in cfg.keys() if k not in valid_keys])
    if unknown:
        raise ValueError(f"Unknown keys in config {pre_args.config}: {unknown}")
    parser.set_defaults(**cfg)
    return parser.parse_args(argv)


def compute_min_dists_to_cloud(
    positions: np.ndarray, cloud_xyz: np.ndarray, chunk_size: int = 50000
) -> np.ndarray:
    """Compute per-position nearest distance to point cloud (chunked brute force)."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [N,3], got {positions.shape}")
    if cloud_xyz.ndim != 2 or cloud_xyz.shape[1] != 3:
        raise ValueError(f"cloud_xyz must be [M,3], got {cloud_xyz.shape}")
    if cloud_xyz.shape[0] == 0:
        return np.full((positions.shape[0],), np.inf, dtype=np.float32)

    min_d2 = np.full((positions.shape[0],), np.inf, dtype=np.float64)
    for st in range(0, cloud_xyz.shape[0], chunk_size):
        ed = min(st + chunk_size, cloud_xyz.shape[0])
        chunk = cloud_xyz[st:ed]  # [K,3]
        # [N,K,3] -> [N,K]
        d2 = np.sum((positions[:, None, :] - chunk[None, :, :]) ** 2, axis=2)
        min_d2 = np.minimum(min_d2, np.min(d2, axis=1))
    return np.sqrt(min_d2).astype(np.float32)


def run_collision_check(
    waypoints: np.ndarray, cloud_xyz: np.ndarray, collision_radius: float
) -> None:
    """Print collision statistics against point cloud by radius threshold."""
    pos = waypoints[:, 1:4].astype(np.float32)
    min_d = compute_min_dists_to_cloud(pos, cloud_xyz)
    min_clearance = float(np.min(min_d))
    coll_mask = min_d <= float(collision_radius)

    print(
        f"[collision] threshold={collision_radius:.3f} m, "
        f"min_clearance={min_clearance:.3f} m"
    )
    if not np.any(coll_mask):
        print("[collision] result=PASS (no waypoint in collision)")
        return

    hit_idx = int(np.argmax(coll_mask))
    hit_t = float(waypoints[hit_idx, 0])
    hit_pos = waypoints[hit_idx, 1:4]
    hit_dist = float(min_d[hit_idx])
    print(
        "[collision] result=COLLISION "
        f"count={int(np.sum(coll_mask))}/{len(coll_mask)} "
        f"first_hit_idx={hit_idx} t={hit_t:.3f}s "
        f"pos=[{hit_pos[0]:.3f}, {hit_pos[1]:.3f}, {hit_pos[2]:.3f}] "
        f"nearest_dist={hit_dist:.3f} m"
    )


def make_path_msg(waypoints: np.ndarray, frame_id: str, rospy, Path, PoseStamped):
    p = Path()
    p.header.frame_id = frame_id
    p.header.stamp = rospy.Time.now()
    for row in waypoints:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = float(row[1])
        pose.pose.position.y = float(row[2])
        pose.pose.position.z = float(row[3])
        pose.pose.orientation.w = 1.0
        p.poses.append(pose)
    return p


def make_drone_marker(frame_id: str, Marker):
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = "collect_replay"
    m.id = 0
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.scale.x = 0.25
    m.scale.y = 0.25
    m.scale.z = 0.25
    m.color.a = 1.0
    m.color.r = 1.0
    m.color.g = 0.2
    m.color.b = 0.1
    m.pose.orientation.w = 1.0
    return m


def main():
    args = parse_args()
    if args.scene_id is None or args.traj_id is None:
        raise ValueError("scene_id and traj_id must be provided via config or CLI.")
    out_size = resolve_video_size(args.img_width, args.img_height)

    paths = build_sample_paths(args.dataset_root, args.scene_id, args.traj_id)
    for key in ("ply", "waypoints", "depths"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"Missing required file: {paths[key]}")

    waypoints = np.load(paths["waypoints"]).astype(np.float32)
    depths = np.load(paths["depths"])
    cloud_xyz = load_ply_xyz(paths["ply"])

    if args.check_collision:
        run_collision_check(waypoints, cloud_xyz, args.collision_radius)

    if args.video_out:
        os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
        export_depth_video(depths, args.video_out, args.rate, out_size=out_size)
        print(f"[replay] depth video saved: {args.video_out}")

    if args.rgb_video_out:
        if not os.path.exists(paths["rgb_video"]):
            raise FileNotFoundError(
                f"Missing RGB file: {paths['rgb_video']} (re-collect dataset with rgb.mp4)"
            )
        rgbs = load_rgb_video(paths["rgb_video"])
        if rgbs.shape[0] != waypoints.shape[0]:
            print(
                "[replay] warning: RGB frame count != waypoint count "
                f"({rgbs.shape[0]} vs {waypoints.shape[0]})"
            )
        os.makedirs(os.path.dirname(args.rgb_video_out) or ".", exist_ok=True)
        export_rgb_video(rgbs, args.rgb_video_out, args.rate, out_size=out_size)
        print(f"[replay] rgb video saved: {args.rgb_video_out}")

    if args.video_only:
        print("[replay] video_only enabled, skip ROS publish")
        return

    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs import point_cloud2
        from std_msgs.msg import Header
        from visualization_msgs.msg import Marker
    except ModuleNotFoundError as e:
        missing = str(e)
        raise RuntimeError(
            "ROS Python dependencies are missing in current environment. "
            "Install with `python -m pip install rospkg catkin_pkg` or switch to "
            "a ROS-ready Python env, or rerun with --video_only."
        ) from e

    rospy.init_node("collect_replay_node", anonymous=True)
    pub_map = rospy.Publisher(args.topic_map, PointCloud2, queue_size=1, latch=True)
    pub_path = rospy.Publisher(args.topic_path, Path, queue_size=1, latch=True)
    pub_marker = rospy.Publisher(args.topic_marker, Marker, queue_size=10)

    header = Header(frame_id=args.frame_id, stamp=rospy.Time.now())
    map_msg = point_cloud2.create_cloud_xyz32(header, cloud_xyz.tolist())
    path_msg = make_path_msg(waypoints, args.frame_id, rospy, Path, PoseStamped)
    pub_map.publish(map_msg)
    pub_path.publish(path_msg)
    print(
        f"[replay] published map({cloud_xyz.shape[0]} pts) + "
        f"path({waypoints.shape[0]} poses)"
    )

    marker = make_drone_marker(args.frame_id, Marker)
    rate = rospy.Rate(max(args.rate, 1.0))

    while not rospy.is_shutdown():
        for i in range(waypoints.shape[0]):
            marker.header.stamp = rospy.Time.now()
            marker.pose.position.x = float(waypoints[i, 1])
            marker.pose.position.y = float(waypoints[i, 2])
            marker.pose.position.z = float(waypoints[i, 3])
            pub_marker.publish(marker)
            rate.sleep()
            if rospy.is_shutdown():
                break
        if not args.loop:
            break

    print("[replay] done")


if __name__ == "__main__":
    main()

