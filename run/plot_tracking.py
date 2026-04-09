#!/usr/bin/env python3
"""Record reference vs actual trajectory during FlowPilot PX4 flights and plot.

For each replanning segment, records:
  - Reference trajectory (evaluated from the PolyTraj polynomial)
  - Actual drone odometry
  - Odom RPY vs differential-flatness RPY

On Ctrl-C or rospy shutdown, saves a multi-panel matplotlib figure.

Usage:
    python plot_tracking.py \
        --odom_topic /some_object_name_vrpn_client/estimated_odometry \
        [--output /tmp/tracking_plot.png]
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import rospy
from nav_msgs.msg import Odometry
from traj_utils.msg import PolyTraj


# ---- geometry helpers -------------------------------------------------------

def quat_to_rpy(w, x, y, z):
    """[w,x,y,z] quaternion -> (roll, pitch, yaw) in degrees."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def flatness_rpy(vel, acc, grav=9.81, dh=0.556, mass=1.5, cp=0.0, veps=1e-6):
    """Compute RPY from differential flatness (matching coords.py)."""
    speed = np.linalg.norm(vel)
    if speed < 1e-3:
        return 0.0, 0.0, 0.0   # hover -> level

    psi = math.atan2(vel[1], vel[0])

    cp_term = math.sqrt(float(vel @ vel) + veps)
    w = (1.0 + cp * cp_term) * vel

    zu = acc + (dh / mass) * w
    zu = zu.copy()
    zu[2] += grav

    zu_norm = np.linalg.norm(zu)
    if zu_norm < 1e-9:
        return 0.0, 0.0, np.degrees(psi)
    z_body = zu / zu_norm

    tilt_den = math.sqrt(2.0 * (1.0 + z_body[2]))
    if tilt_den < 1e-9:
        return 0.0, 0.0, np.degrees(psi)
    tilt0 = 0.5 * tilt_den
    tilt1 = -z_body[1] / tilt_den
    tilt2 = z_body[0] / tilt_den

    c_half = math.cos(0.5 * psi)
    s_half = math.sin(0.5 * psi)
    qw = tilt0 * c_half
    qx = tilt1 * c_half + tilt2 * s_half
    qy = tilt2 * c_half - tilt1 * s_half
    qz = tilt0 * s_half

    return quat_to_rpy(qw, qx, qy, qz)


# ---- polynomial evaluation -------------------------------------------------

def eval_poly(coefs, order, t):
    """Evaluate polynomial:  p(t) = coefs[order]*t^0 + ... + coefs[0]*t^order."""
    val = 0.0
    tn = 1.0
    for i in range(order, -1, -1):
        val += tn * coefs[i]
        tn *= t
    return val


def eval_poly_deriv(coefs, order, t):
    """First derivative of polynomial."""
    val = 0.0
    tn = 1.0
    n = 1
    for i in range(order - 1, -1, -1):
        val += n * tn * coefs[i]
        tn *= t
        n += 1
    return val


def eval_poly_acc(coefs, order, t):
    """Second derivative of polynomial."""
    val = 0.0
    tn = 1.0
    m, n = 1, 2
    for i in range(order - 2, -1, -1):
        val += m * n * tn * coefs[i]
        tn *= t
        m += 1
        n += 1
    return val


# ---- data structures --------------------------------------------------------

@dataclass
class OdomSample:
    t: float        # relative time in segment (seconds)
    pos: np.ndarray  # [3]
    vel: np.ndarray  # [3]
    quat_wxyz: np.ndarray  # [4]
    odom_rpy: np.ndarray   # [3] degrees
    flat_rpy: np.ndarray   # [3] degrees


@dataclass
class Segment:
    """One replanning segment: from PolyTraj publish to the next replanning."""
    traj_id: int
    order: int
    coef_x: np.ndarray
    coef_y: np.ndarray
    coef_z: np.ndarray
    duration: float
    start_wall: float          # time.time() when this segment started
    odom_samples: List[OdomSample] = field(default_factory=list)

    def ref_pos_at(self, t):
        t = max(0.0, min(t, self.duration))
        x = eval_poly(self.coef_x, self.order, t)
        y = eval_poly(self.coef_y, self.order, t)
        z = eval_poly(self.coef_z, self.order, t)
        return np.array([x, y, z])

    def ref_vel_at(self, t):
        t = max(0.0, min(t, self.duration))
        vx = eval_poly_deriv(self.coef_x, self.order, t)
        vy = eval_poly_deriv(self.coef_y, self.order, t)
        vz = eval_poly_deriv(self.coef_z, self.order, t)
        return np.array([vx, vy, vz])

    def ref_acc_at(self, t):
        t = max(0.0, min(t, self.duration))
        ax = eval_poly_acc(self.coef_x, self.order, t)
        ay = eval_poly_acc(self.coef_y, self.order, t)
        az = eval_poly_acc(self.coef_z, self.order, t)
        return np.array([ax, ay, az])


# ---- recorder ---------------------------------------------------------------

class TrackingRecorder:
    def __init__(self, odom_topic: str, traj_topic: str):
        self.segments: List[Segment] = []
        self._cur_seg: Optional[Segment] = None
        self._lock = threading.Lock()

        rospy.Subscriber(traj_topic, PolyTraj, self._traj_cb, queue_size=10)
        rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=200)

    def _traj_cb(self, msg: PolyTraj):
        order = int(msg.order)
        n_pieces = len(msg.duration)
        if n_pieces == 0:
            return
        per = order + 1
        # Take first piece (FlowPilot publishes single-piece trajectories)
        coef_x = np.array(msg.coef_x[:per], dtype=np.float64)
        coef_y = np.array(msg.coef_y[:per], dtype=np.float64)
        coef_z = np.array(msg.coef_z[:per], dtype=np.float64)
        dur = float(msg.duration[0])
        seg = Segment(
            traj_id=int(msg.traj_id), order=order,
            coef_x=coef_x, coef_y=coef_y, coef_z=coef_z,
            duration=dur, start_wall=time.time())
        with self._lock:
            if self._cur_seg is not None:
                self.segments.append(self._cur_seg)
            self._cur_seg = seg

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            seg = self._cur_seg
        if seg is None:
            return

        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        vel = np.array([v.x, v.y, v.z])
        quat = np.array([q.w, q.x, q.y, q.z])

        t_rel = time.time() - seg.start_wall

        odom_rpy = np.array(quat_to_rpy(q.w, q.x, q.y, q.z))

        # Flatness RPY from vel + trajectory reference acc
        ref_acc = seg.ref_acc_at(t_rel)
        flat_r, flat_p, flat_y = flatness_rpy(vel, ref_acc)
        flat_rpy_arr = np.array([flat_r, flat_p, flat_y])

        sample = OdomSample(t=t_rel, pos=pos, vel=vel, quat_wxyz=quat,
                            odom_rpy=odom_rpy, flat_rpy=flat_rpy_arr)
        with self._lock:
            if self._cur_seg is not None:
                self._cur_seg.odom_samples.append(sample)

    def finalize(self):
        with self._lock:
            if self._cur_seg is not None:
                self.segments.append(self._cur_seg)
                self._cur_seg = None


# ---- plotting ---------------------------------------------------------------

def plot_tracking(segments: List[Segment], output_path: str):
    if not segments:
        print('[plot] no segments recorded.')
        return

    # Filter out segments with too few samples
    segments = [s for s in segments if len(s.odom_samples) >= 5]
    if not segments:
        print('[plot] no segments with enough samples.')
        return

    n_seg = len(segments)
    fig = plt.figure(figsize=(20, 5 * n_seg + 6))
    gs = GridSpec(n_seg + 2, 3, figure=fig, hspace=0.35, wspace=0.30)

    all_ref_x, all_ref_y = [], []
    all_act_x, all_act_y = [], []
    all_pos_err = []
    all_pos_err_t = []
    t_offset = 0.0

    for i, seg in enumerate(segments):
        samples = seg.odom_samples
        ts = np.array([s.t for s in samples])
        act_pos = np.array([s.pos for s in samples])
        odom_rpy = np.array([s.odom_rpy for s in samples])
        flat_rpy = np.array([s.flat_rpy for s in samples])

        # Evaluate reference at odom timestamps
        ref_pos = np.array([seg.ref_pos_at(t) for t in ts])
        pos_err = np.linalg.norm(ref_pos - act_pos, axis=1)

        # Collect for overview
        ref_ts_dense = np.linspace(0, seg.duration, 100)
        ref_dense = np.array([seg.ref_pos_at(t) for t in ref_ts_dense])
        all_ref_x.extend(ref_dense[:, 0])
        all_ref_y.extend(ref_dense[:, 1])
        all_act_x.extend(act_pos[:, 0])
        all_act_y.extend(act_pos[:, 1])
        all_pos_err.extend(pos_err)
        all_pos_err_t.extend(ts + t_offset)
        if len(ts) > 0:
            t_offset += ts[-1]

        # Row i: [XY traj, pos error, RPY comparison]
        # -- XY trajectory
        ax_xy = fig.add_subplot(gs[i, 0])
        ax_xy.plot(ref_dense[:, 0], ref_dense[:, 1], 'b-', lw=1.5, label='reference')
        ax_xy.plot(act_pos[:, 0], act_pos[:, 1], 'r-', lw=1, alpha=0.8, label='actual')
        ax_xy.plot(act_pos[0, 0], act_pos[0, 1], 'go', ms=6, label='start')
        ax_xy.set_title(f'Seg {i} (traj_id={seg.traj_id})', fontsize=10)
        ax_xy.set_xlabel('x [m]')
        ax_xy.set_ylabel('y [m]')
        ax_xy.set_aspect('equal')
        ax_xy.legend(fontsize=7)
        ax_xy.grid(True, alpha=0.3)

        # -- Position error vs time
        ax_err = fig.add_subplot(gs[i, 1])
        ax_err.plot(ts, pos_err, 'k-', lw=1)
        ax_err.set_xlabel('t [s]')
        ax_err.set_ylabel('pos error [m]')
        ax_err.set_title(f'Tracking error (mean={np.mean(pos_err):.3f}m)', fontsize=10)
        ax_err.grid(True, alpha=0.3)

        # -- RPY comparison
        ax_rpy = fig.add_subplot(gs[i, 2])
        for j, (label, color) in enumerate(zip(['roll', 'pitch', 'yaw'],
                                                ['r', 'g', 'b'])):
            ax_rpy.plot(ts, odom_rpy[:, j], color + '-', lw=1,
                        label=f'odom {label}', alpha=0.8)
            ax_rpy.plot(ts, flat_rpy[:, j], color + '--', lw=1,
                        label=f'flat {label}', alpha=0.8)
        ax_rpy.set_xlabel('t [s]')
        ax_rpy.set_ylabel('angle [deg]')
        ax_rpy.set_title('Odom RPY vs Flatness RPY', fontsize=10)
        ax_rpy.legend(fontsize=6, ncol=2)
        ax_rpy.grid(True, alpha=0.3)

    # Bottom row 1: overview XY
    ax_ov = fig.add_subplot(gs[n_seg, :2])
    ax_ov.plot(all_ref_x, all_ref_y, 'b-', lw=1.2, alpha=0.6, label='reference')
    ax_ov.plot(all_act_x, all_act_y, 'r-', lw=1, alpha=0.6, label='actual')
    # Mark replanning boundaries
    for i, seg in enumerate(segments):
        if seg.odom_samples:
            p = seg.odom_samples[0].pos
            ax_ov.plot(p[0], p[1], 'k^', ms=4)
    ax_ov.set_title('Full flight: reference vs actual (triangles = replan)', fontsize=11)
    ax_ov.set_xlabel('x [m]')
    ax_ov.set_ylabel('y [m]')
    ax_ov.set_aspect('equal')
    ax_ov.legend(fontsize=8)
    ax_ov.grid(True, alpha=0.3)

    # Bottom row 2: position error over full flight
    ax_err_all = fig.add_subplot(gs[n_seg, 2])
    ax_err_all.plot(all_pos_err_t, all_pos_err, 'k-', lw=0.8)
    ax_err_all.set_title(f'Tracking error (overall mean={np.mean(all_pos_err):.3f}m)', fontsize=10)
    ax_err_all.set_xlabel('t [s]')
    ax_err_all.set_ylabel('pos error [m]')
    ax_err_all.grid(True, alpha=0.3)

    # Bottom row 3: altitude comparison
    ax_z = fig.add_subplot(gs[n_seg + 1, :])
    t_acc = 0.0
    for seg in segments:
        samples = seg.odom_samples
        if not samples:
            continue
        ts = np.array([s.t for s in samples]) + t_acc
        act_z = np.array([s.pos[2] for s in samples])
        ref_z = np.array([seg.ref_pos_at(s.t)[2] for s in samples])
        ax_z.plot(ts, ref_z, 'b-', lw=1, alpha=0.6)
        ax_z.plot(ts, act_z, 'r-', lw=1, alpha=0.6)
        ax_z.axvline(ts[0], color='gray', ls='--', lw=0.5, alpha=0.5)
        if len(samples) > 0:
            t_acc += samples[-1].t
    ax_z.set_title('Altitude: reference (blue) vs actual (red), dashed = replan', fontsize=10)
    ax_z.set_xlabel('t [s]')
    ax_z.set_ylabel('z [m]')
    ax_z.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] saved to {output_path}')

    # Save raw data alongside the plot
    data_path = output_path.rsplit('.', 1)[0] + '_data.npz'
    seg_data = {}
    for i, seg in enumerate(segments):
        samples = seg.odom_samples
        if not samples:
            continue
        ts = np.array([s.t for s in samples])
        act_pos = np.array([s.pos for s in samples])
        act_vel = np.array([s.vel for s in samples])
        odom_rpy = np.array([s.odom_rpy for s in samples])
        flat_rpy = np.array([s.flat_rpy for s in samples])
        ref_pos = np.array([seg.ref_pos_at(t) for t in ts])
        ref_vel = np.array([seg.ref_vel_at(t) for t in ts])
        seg_data[f'seg{i:03d}_t']        = ts
        seg_data[f'seg{i:03d}_act_pos']  = act_pos
        seg_data[f'seg{i:03d}_act_vel']  = act_vel
        seg_data[f'seg{i:03d}_ref_pos']  = ref_pos
        seg_data[f'seg{i:03d}_ref_vel']  = ref_vel
        seg_data[f'seg{i:03d}_odom_rpy'] = odom_rpy
        seg_data[f'seg{i:03d}_flat_rpy'] = flat_rpy
        seg_data[f'seg{i:03d}_traj_id']  = np.array([seg.traj_id])
        seg_data[f'seg{i:03d}_duration'] = np.array([seg.duration])
        seg_data[f'seg{i:03d}_coefs']    = np.stack([seg.coef_x, seg.coef_y, seg.coef_z])
    np.savez_compressed(data_path, **seg_data)
    print(f'[data] saved to {data_path}')


# ---- main -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Record and plot tracking performance.')
    p.add_argument('--odom_topic', type=str,
                   default='/some_object_name_vrpn_client/estimated_odometry')
    p.add_argument('--traj_topic', type=str,
                   default='/drone_0_planning/trajectory')
    p.add_argument('--output', type=str, default='/tmp/tracking_plot.png')
    return p.parse_args()


def main():
    args = parse_args()
    rospy.init_node('plot_tracking', anonymous=True)
    recorder = TrackingRecorder(args.odom_topic, args.traj_topic)
    print(f'[plot_tracking] recording... odom={args.odom_topic}  traj={args.traj_topic}')
    print(f'[plot_tracking] Ctrl-C to stop and generate plot.')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

    recorder.finalize()
    print(f'[plot_tracking] {len(recorder.segments)} segments recorded.')
    plot_tracking(recorder.segments, args.output)


if __name__ == '__main__':
    main()
