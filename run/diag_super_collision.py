#!/usr/bin/env python3
"""
Diagnostic: does SUPER actually fly collision-free on long-range goals?

Prerequisites (run inside Docker):
  1. roscore already running
  2. SUPER click_demo.launch already running with its default PCD/config
  3. conda activate yopo  (for scipy)

Usage:
  python diag_super_collision.py

Output:
  /tmp/diag_odom_all.npy   -- all recorded odom positions  [N, 3]
  /tmp/diag_odom_goals.npy -- per-goal odom arrays         list of [M, 3]
  Prints collision summary.
"""

import threading
import time

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

# ---------------------------------------------------------------------------
# Goals are generated at runtime relative to the drone's init position:
# 4 cardinal directions × GOAL_DIST metres, all at FLIGHT_Z height.
# ---------------------------------------------------------------------------
GOAL_DIST    = 18.0    # m  — distance of each test goal from init pos
FLIGHT_Z     = 1.5     # m  — fixed flight height

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
GOAL_REACH_DIST = 4.0      # m  — consider goal reached
GOAL_TIMEOUT    = 90.0     # s  — give up on this goal
COLLISION_R     = 0.3      # m  — robot radius for collision check
SUPER_CFG       = "/root/workspace/ROS/src/SUPER/super_planner/config/click_smooth_ros1.yaml"


# ---------------------------------------------------------------------------
class OdomRecorder:
    def __init__(self):
        self._lock    = threading.Lock()
        self._cur     = None      # latest [x,y,z]
        self._history = []        # list of np.array [x,y,z]

    def callback(self, msg):
        p = msg.pose.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        with self._lock:
            self._cur = pos
            self._history.append(pos)

    def position(self):
        with self._lock:
            return None if self._cur is None else self._cur.copy()

    def clear(self):
        with self._lock:
            self._history.clear()

    def snapshot(self):
        with self._lock:
            return list(self._history)


def send_goal(pub, goal: np.ndarray):
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.header.stamp    = rospy.Time.now()
    msg.pose.position.x = float(goal[0])
    msg.pose.position.y = float(goal[1])
    msg.pose.position.z = float(goal[2])
    msg.pose.orientation.w = 1.0
    pub.publish(msg)


def load_pcd(path: str) -> np.ndarray:
    """Load XYZ points from a PCD file (ASCII or binary). Returns [N,3] float64."""
    import struct

    with open(path, "rb") as f:
        raw = f.read()

    # Parse header
    header_end = raw.find(b"\nDATA ")
    header_txt = raw[:header_end].decode("ascii", errors="ignore")
    data_line  = raw[header_end + 1:].split(b"\n", 1)
    data_type  = data_line[0].split()[1].decode().strip()   # ascii / binary / binary_compressed
    body       = data_line[1]

    fields, sizes, types, count, width, height = [], [], [], [], 0, 0
    for line in header_txt.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "FIELDS":
            fields = tok[1:]
        elif tok[0] == "SIZE":
            sizes = [int(s) for s in tok[1:]]
        elif tok[0] == "TYPE":
            types = tok[1:]
        elif tok[0] == "COUNT":
            count = [int(c) for c in tok[1:]]
        elif tok[0] == "WIDTH":
            width = int(tok[1])
        elif tok[0] == "HEIGHT":
            height = int(tok[1])

    n_pts = width * height

    # Build struct format for one point
    fmt_map = {"F": {4: "f", 8: "d"}, "I": {4: "i"}, "U": {4: "I"}}
    point_fmt = "<"
    point_size = 0
    xyz_offsets = {}
    off = 0
    for i, (f, s, t, c) in enumerate(zip(fields, sizes, types, count)):
        ch = fmt_map.get(t, {}).get(s, f"x" * s)
        for ci in range(c):
            if f in ("x", "y", "z") and ci == 0:
                xyz_offsets[f] = (off, ch)
            point_fmt  += ch
            point_size += s
        off += s * c

    if data_type == "ascii":
        lines = body.decode("ascii", errors="ignore").split("\n")
        pts = []
        for line in lines:
            tok = line.split()
            if len(tok) < 3:
                continue
            try:
                x = float(tok[fields.index("x")])
                y = float(tok[fields.index("y")])
                z = float(tok[fields.index("z")])
                pts.append([x, y, z])
            except (ValueError, IndexError):
                continue
        return np.array(pts, dtype=np.float64)

    elif data_type == "binary":
        pts = np.zeros((n_pts, 3), dtype=np.float64)
        xi = fields.index("x"); yi = fields.index("y"); zi = fields.index("z")
        for i in range(n_pts):
            chunk = body[i * point_size:(i + 1) * point_size]
            vals  = struct.unpack(point_fmt, chunk)
            pts[i] = [vals[xi], vals[yi], vals[zi]]
        return pts

    elif data_type == "binary_compressed":
        import zlib
        compressed_size   = struct.unpack_from("<I", body, 0)[0]
        uncompressed_size = struct.unpack_from("<I", body, 4)[0]
        comp_data = body[8: 8 + compressed_size]
        raw_data  = zlib.decompress(comp_data, wbits=-15)   # raw deflate

        # binary_compressed stores each field column-contiguously
        pts = np.zeros((n_pts, 3), dtype=np.float64)
        col_off = 0
        for fi, (fn, fs, ft, fc) in enumerate(zip(fields, sizes, types, count)):
            col_bytes = fs * fc * n_pts
            if fn in ("x", "y", "z"):
                dt = np.dtype("<f4") if fs == 4 else np.dtype("<f8")
                col = np.frombuffer(raw_data[col_off: col_off + col_bytes], dtype=dt)
                idx = ("x", "y", "z").index(fn)
                pts[:, idx] = col
            col_off += col_bytes
        return pts

    else:
        raise ValueError(f"Unknown PCD data type: {data_type}")


def check_collision_pcd(positions: list, pcd_path: str, radius: float):
    """Return list of (idx, dist, pos) for positions within `radius` of PCD."""
    from scipy.spatial import cKDTree

    pts = load_pcd(pcd_path)
    if len(pts) == 0:
        print("[WARN] PCD is empty.")
        return []

    print(f"[diag] Loaded {len(pts)} points from PCD.")
    tree = cKDTree(pts)
    hits = []
    for i, pos in enumerate(positions):
        d, _ = tree.query(pos)
        if d < radius:
            hits.append((i, float(d), pos))
    return hits


# ---------------------------------------------------------------------------
def main():
    rospy.init_node("diag_super_collision", anonymous=True)

    recorder = OdomRecorder()
    rospy.Subscriber("/lidar_slam/odom", Odometry, recorder.callback)
    goal_pub = rospy.Publisher("/goal", PoseStamped, queue_size=1)

    # ---- Read PCD path from SUPER config ----
    with open(SUPER_CFG) as f:
        cfg = yaml.safe_load(f)
    pcd_path = cfg["rog_map"]["pcd_name"]
    print(f"[diag] PCD used by SUPER: {pcd_path}")

    # ---- Wait for odometry ----
    print("[diag] Waiting for /lidar_slam/odom ...")
    t0 = time.time()
    while recorder.position() is None:
        if time.time() - t0 > 30:
            print("[diag] ERROR: no odometry after 30 s — is click_demo.launch running?")
            return
        time.sleep(0.1)

    init_pos = recorder.position()
    print(f"[diag] Init pos: {init_pos.round(2)}")

    # Build goals: 4 cardinal directions × GOAL_DIST from init, fixed z
    GOALS = [
        init_pos + np.array([ GOAL_DIST,       0.0, 0.0]),
        init_pos + np.array([-GOAL_DIST,       0.0, 0.0]),
        init_pos + np.array([       0.0,  GOAL_DIST, 0.0]),
        init_pos + np.array([       0.0, -GOAL_DIST, 0.0]),
    ]
    for g in GOALS:
        g[2] = FLIGHT_Z

    for i, g in enumerate(GOALS):
        d = np.linalg.norm(np.array(g) - init_pos)
        print(f"  Goal {i}: {g}  dist={d:.1f} m")

    # ---- Fly each goal ----
    all_odom   = []   # all positions across all goals
    goal_odems = []   # per-goal position lists

    for gi, goal_raw in enumerate(GOALS):
        goal = np.array(goal_raw, dtype=np.float64)
        dist_init = np.linalg.norm(goal - init_pos)
        print(f"\n[Goal {gi+1}/{len(GOALS)}] {goal.tolist()}  dist_from_init={dist_init:.1f}m")

        recorder.clear()
        send_goal(goal_pub, goal)

        t0      = time.time()
        reached = False
        while time.time() - t0 < GOAL_TIMEOUT:
            pos = recorder.position()
            if pos is not None:
                d = np.linalg.norm(pos - goal)
                elapsed = time.time() - t0
                print(f"  t={elapsed:5.1f}s  pos={pos.round(1)}  dist_to_goal={d:.1f}m", end="\r")
                if d < GOAL_REACH_DIST:
                    print()
                    print(f"  -> REACHED  (dist={d:.2f}m, t={elapsed:.1f}s)")
                    reached = True
                    break
            time.sleep(0.1)

        if not reached:
            print()
            print(f"  -> TIMEOUT after {GOAL_TIMEOUT}s")

        hist = recorder.snapshot()
        print(f"  Recorded {len(hist)} odom points")
        goal_odems.append(np.array(hist) if hist else np.zeros((0, 3)))
        all_odom.extend(hist)

    # ---- Save odom ----
    all_arr = np.array(all_odom) if all_odom else np.zeros((0, 3))
    np.save("/tmp/diag_odom_all.npy", all_arr)
    print(f"\n[diag] Saved {len(all_odom)} odom pts to /tmp/diag_odom_all.npy")

    # ---- Collision check ----
    print(f"\n=== Collision Check (radius={COLLISION_R}m, PCD={pcd_path}) ===")
    if len(all_odom) == 0:
        print("No odom recorded — nothing to check.")
        return

    hits = check_collision_pcd(all_odom, pcd_path, COLLISION_R)
    total = len(all_odom)

    if hits:
        print(f"COLLISIONS: {len(hits)} / {total} points ({100*len(hits)/total:.1f}%)")
        print("First 10 collision points:")
        for idx, d, pos in hits[:10]:
            print(f"  odom_idx={idx:5d}  dist_to_obstacle={d:.3f}m  pos={pos.round(2)}")
    else:
        print(f"No collisions detected ({total} points, all clear at r={COLLISION_R}m)")

    # Per-goal summary
    print("\nPer-goal summary:")
    base = 0
    for gi, arr in enumerate(goal_odems):
        if len(arr) == 0:
            print(f"  Goal {gi+1}: no data")
            continue
        goal_hits = [h for h in hits if base <= h[0] < base + len(arr)]
        print(f"  Goal {gi+1}: {len(arr):4d} pts, {len(goal_hits)} collisions"
              + (f"  first at idx={goal_hits[0][0]-base}, dist={goal_hits[0][1]:.3f}m" if goal_hits else ""))
        base += len(arr)


if __name__ == "__main__":
    main()
