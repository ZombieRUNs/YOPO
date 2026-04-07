#!/usr/bin/env python3
"""
Passive bridge validator for:

  PX4 + Gazebo + MAVROS
          |
          v
  /mavros/local_position/odom
          |
          v
    flightros_node + Flightmare
          |
          v
  /RGB_image, /depth_image, /camera_info

This script does not publish any control commands. It only subscribes to the
relevant topics, reports their rates/freshness, and can optionally save one RGB
snapshot and one depth snapshot for quick inspection.
"""

import argparse
import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image


class TopicMonitor:
    def __init__(self, name: str, max_samples: int = 200) -> None:
        self.name = name
        self.arrivals = deque(maxlen=max_samples)
        self.count = 0
        self.last_wall_time: Optional[float] = None
        self.last_header_stamp: Optional[rospy.Time] = None

    def note(self, header_stamp: Optional[rospy.Time] = None) -> None:
        now = time.time()
        self.arrivals.append(now)
        self.count += 1
        self.last_wall_time = now
        if header_stamp is not None and header_stamp != rospy.Time():
            self.last_header_stamp = header_stamp

    def hz(self) -> float:
        if len(self.arrivals) < 2:
            return 0.0
        dt = self.arrivals[-1] - self.arrivals[0]
        if dt <= 1e-6:
            return 0.0
        return (len(self.arrivals) - 1) / dt

    def freshness(self) -> Optional[float]:
        if self.last_wall_time is None:
            return None
        return max(0.0, time.time() - self.last_wall_time)


class BridgeValidator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bridge = CvBridge()

        self.odom_mon = TopicMonitor(args.odom_topic)
        self.state_mon = TopicMonitor(args.state_topic)
        self.rgb_mon = TopicMonitor(args.rgb_topic)
        self.depth_mon = TopicMonitor(args.depth_topic)
        self.cam_mon = TopicMonitor(args.camera_info_topic)

        self.last_state: Optional[State] = None
        self.last_odom: Optional[Odometry] = None
        self.last_rgb_shape = None
        self.last_depth_shape = None
        self.last_cam_info: Optional[CameraInfo] = None
        self.odom_history = deque()

        self.save_dir = Path(args.save_dir).resolve() if args.save_dir else None
        self.saved_rgb = False
        self.saved_depth = False
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        rospy.Subscriber(args.odom_topic, Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber(args.state_topic, State, self.state_cb, queue_size=1)
        rospy.Subscriber(args.rgb_topic, Image, self.rgb_cb, queue_size=1)
        rospy.Subscriber(args.depth_topic, Image, self.depth_cb, queue_size=1)
        rospy.Subscriber(args.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)

    def odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg
        self.odom_mon.note(msg.header.stamp)
        pose = msg.pose.pose.position
        twist = msg.twist.twist.linear
        now = time.time()
        self.odom_history.append((now, pose.x, pose.y, pose.z, twist.x, twist.y, twist.z))
        self.prune_odom_history(now)

    def state_cb(self, msg: State) -> None:
        self.last_state = msg
        self.state_mon.note()

    def rgb_cb(self, msg: Image) -> None:
        self.rgb_mon.note(msg.header.stamp)
        self.last_rgb_shape = (msg.height, msg.width, msg.encoding)
        if self.save_dir is None or self.saved_rgb:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("RGB snapshot conversion failed: %s", exc)
            return
        out_path = self.save_dir / "rgb_snapshot.png"
        cv2.imwrite(str(out_path), image)
        rospy.loginfo("Saved RGB snapshot to %s", out_path)
        self.saved_rgb = True

    def depth_cb(self, msg: Image) -> None:
        self.depth_mon.note(msg.header.stamp)
        self.last_depth_shape = (msg.height, msg.width, msg.encoding)
        if self.save_dir is None or self.saved_depth:
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logwarn("Depth snapshot conversion failed: %s", exc)
            return

        depth = np.asarray(depth, dtype=np.float32)
        np.save(self.save_dir / "depth_snapshot.npy", depth)

        finite = np.isfinite(depth)
        preview = np.zeros_like(depth, dtype=np.uint8)
        if np.any(finite):
            finite_depth = np.clip(depth[finite], 0.0, float(self.args.depth_clip_max))
            preview_vals = (finite_depth / float(self.args.depth_clip_max) * 255.0).astype(np.uint8)
            preview[finite] = preview_vals
        cv2.imwrite(str(self.save_dir / "depth_snapshot_preview.png"), preview)
        rospy.loginfo("Saved depth snapshot to %s", self.save_dir)
        self.saved_depth = True

    def camera_info_cb(self, msg: CameraInfo) -> None:
        self.last_cam_info = msg
        self.cam_mon.note(msg.header.stamp)

    def prune_odom_history(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        window = max(0.5, float(self.args.motion_window))
        while self.odom_history and (now - self.odom_history[0][0]) > window:
            self.odom_history.popleft()

    def stamp_delta(self, mon: TopicMonitor) -> Optional[float]:
        if self.odom_mon.last_header_stamp is None or mon.last_header_stamp is None:
            return None
        return abs((mon.last_header_stamp - self.odom_mon.last_header_stamp).to_sec())

    def pose_desc(self) -> str:
        if self.last_odom is None:
            return "pose=missing"
        position = self.last_odom.pose.pose.position
        velocity = self.last_odom.twist.twist.linear
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        return (
            f"pose=({position.x:.2f},{position.y:.2f},{position.z:.2f})m "
            f"vel=({velocity.x:.2f},{velocity.y:.2f},{velocity.z:.2f})m/s "
            f"speed={speed:.2f}m/s"
        )

    def motion_desc(self) -> str:
        self.prune_odom_history()
        if len(self.odom_history) < 2:
            return "motion(window=incomplete)"

        xs = [sample[1] for sample in self.odom_history]
        ys = [sample[2] for sample in self.odom_history]
        zs = [sample[3] for sample in self.odom_history]
        window_sec = self.odom_history[-1][0] - self.odom_history[0][0]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        z_span = max(zs) - min(zs)
        xy_span = math.sqrt(x_span * x_span + y_span * y_span)

        if xy_span < 0.20 and z_span < 0.15:
            motion_hint = "hover-like"
        elif xy_span >= 0.50 and z_span < 0.30:
            motion_hint = "planar-track-like"
        else:
            motion_hint = "transitional"

        return (
            f"motion(window={window_sec:.1f}s, x_span={x_span:.2f}m, y_span={y_span:.2f}m, "
            f"z_span={z_span:.2f}m, xy_span={xy_span:.2f}m, hint={motion_hint})"
        )

    def report(self) -> None:
        state_desc = "state=missing"
        if self.last_state is not None:
            state_desc = f'state=connected:{self.last_state.connected} armed:{self.last_state.armed} mode:{self.last_state.mode}'

        odom_desc = f"odom={self.odom_mon.hz():5.1f}Hz"
        rgb_desc = f"rgb={self.rgb_mon.hz():5.1f}Hz"
        depth_desc = f"depth={self.depth_mon.hz():5.1f}Hz"
        cam_desc = f"cam={self.cam_mon.hz():5.1f}Hz"

        odom_fresh = self.odom_mon.freshness()
        rgb_fresh = self.rgb_mon.freshness()
        depth_fresh = self.depth_mon.freshness()

        delta_rgb = self.stamp_delta(self.rgb_mon)
        delta_depth = self.stamp_delta(self.depth_mon)
        pose_desc = self.pose_desc()
        motion_desc = self.motion_desc()

        fresh_desc = (
            f"freshness(odom={odom_fresh:.2f}s, rgb={rgb_fresh:.2f}s, depth={depth_fresh:.2f}s)"
            if None not in (odom_fresh, rgb_fresh, depth_fresh)
            else "freshness(incomplete)"
        )
        delta_desc = (
            f"stamp_delta(rgb={delta_rgb:.3f}s, depth={delta_depth:.3f}s)"
            if None not in (delta_rgb, delta_depth)
            else "stamp_delta(incomplete)"
        )

        shape_desc = []
        if self.last_rgb_shape is not None:
            shape_desc.append(f"rgb_shape={self.last_rgb_shape}")
        if self.last_depth_shape is not None:
            shape_desc.append(f"depth_shape={self.last_depth_shape}")
        if self.last_cam_info is not None:
            shape_desc.append(f"camera_info=({self.last_cam_info.width}x{self.last_cam_info.height})")

        rospy.loginfo(
            "%s | %s | %s | %s | %s | %s | %s | %s | %s",
            state_desc,
            odom_desc,
            rgb_desc,
            depth_desc,
            cam_desc,
            fresh_desc,
            delta_desc,
            pose_desc,
            motion_desc,
        )
        if shape_desc:
            rospy.loginfo("%s", " | ".join(shape_desc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the PX4 -> Flightmare render bridge.")
    parser.add_argument("--odom-topic", default="/mavros/local_position/odom")
    parser.add_argument("--state-topic", default="/mavros/state")
    parser.add_argument("--rgb-topic", default="/RGB_image")
    parser.add_argument("--depth-topic", default="/depth_image")
    parser.add_argument("--camera-info-topic", default="/camera_info")
    parser.add_argument("--report-period", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds; 0 means run until Ctrl+C.")
    parser.add_argument("--save-dir", default="", help="Optional directory for saving one RGB/depth snapshot.")
    parser.add_argument("--depth-clip-max", type=float, default=20.0)
    parser.add_argument("--motion-window", type=float, default=5.0, help="Sliding window in seconds for motion span summary.")
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    rospy.init_node("test_px4_flightmare_bridge", anonymous=False)
    args = parse_args()
    validator = BridgeValidator(args)

    rospy.loginfo("Bridge validator started.")
    rospy.loginfo("Watching odom=%s rgb=%s depth=%s", args.odom_topic, args.rgb_topic, args.depth_topic)
    if args.save_dir:
        rospy.loginfo("Snapshots will be saved under %s", Path(args.save_dir).resolve())

    start_time = time.time()
    rate_hz = max(0.2, 1.0 / max(0.1, float(args.report_period)))
    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        validator.report()
        if args.duration > 0.0 and (time.time() - start_time) >= args.duration:
            rospy.loginfo("Duration reached, exiting bridge validator.")
            break
        rate.sleep()


if __name__ == "__main__":
    main()
