#!/usr/bin/env python3
import rosbag
import numpy as np
import argparse

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def pose_to_vec(msg):
    return np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z
    ])


def odom_to_vec(msg):
    return np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    ])


def compute_errors(bag_path):
    bag = rosbag.Bag(bag_path)

    ref_times = []
    ref_positions = []

    raw_times = []
    raw_positions = []

    ekf_times = []
    ekf_positions = []

    # --- Read bag ---
    for topic, msg, t in bag.read_messages():
        time = t.to_sec()

        if topic == "/hummingbird/command/pose":
            ref_times.append(time)
            ref_positions.append(pose_to_vec(msg))

        elif topic == "/hummingbird/odometry_sensor1/odometry":
            raw_times.append(time)
            raw_positions.append(odom_to_vec(msg))

        elif topic == "/hummingbird/ekf/odometry":
            ekf_times.append(time)
            ekf_positions.append(odom_to_vec(msg))

    bag.close()

    ref_times = np.array(ref_times)
    ref_positions = np.array(ref_positions)

    raw_times = np.array(raw_times)
    raw_positions = np.array(raw_positions)

    ekf_times = np.array(ekf_times)
    ekf_positions = np.array(ekf_positions)

    # --- Interpolate reference to match timestamps ---
    def interpolate(ref_t, ref_p, target_t):
        interp = np.zeros((len(target_t), 3))
        for i in range(3):
            interp[:, i] = np.interp(target_t, ref_t, ref_p[:, i])
        return interp

    ref_on_raw = interpolate(ref_times, ref_positions, raw_times)
    ref_on_ekf = interpolate(ref_times, ref_positions, ekf_times)

    # --- Compute errors ---
    raw_error = raw_positions - ref_on_raw
    ekf_error = ekf_positions - ref_on_ekf

    raw_norm = np.linalg.norm(raw_error, axis=1)
    ekf_norm = np.linalg.norm(ekf_error, axis=1)

    # --- Metrics ---
    def stats(err):
        return {
            "RMS": np.sqrt(np.mean(err**2)),
            "Mean": np.mean(err),
            "Max": np.max(err)
        }

    raw_stats = stats(raw_norm)
    ekf_stats = stats(ekf_norm)

    return raw_stats, ekf_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="Path to rosbag")
    args = parser.parse_args()

    raw_stats, ekf_stats = compute_errors(args.bag)

    print("\n===== TRACKING ERROR COMPARISON =====\n")

    print("RAW ODOMETRY:")
    for k, v in raw_stats.items():
        print(f"  {k}: {v:.4f} m")

    print("\nEKF ODOMETRY:")
    for k, v in ekf_stats.items():
        print(f"  {k}: {v:.4f} m")

    print("\n=====================================\n")


if __name__ == "__main__":
    main()
