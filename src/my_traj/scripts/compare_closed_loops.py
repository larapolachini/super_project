#!/usr/bin/env python3
import argparse
import math
import sys

import numpy as np
import rosbag


REF_TOPIC = "/hummingbird/command/pose"
RAW_ODOM_TOPIC = "/hummingbird/odometry_sensor1/odometry"
EKF_ODOM_TOPIC = "/hummingbird/ekf/odometry"


def pose_to_vec(msg):
    return np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z
    ], dtype=float)


def odom_to_vec(msg):
    return np.array([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    ], dtype=float)


def read_bag_positions(bag_path, odom_topic):
    ref_times = []
    ref_positions = []

    odom_times = []
    odom_positions = []

    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[REF_TOPIC, odom_topic]):
            ts = t.to_sec()

            if topic == REF_TOPIC:
                ref_times.append(ts)
                ref_positions.append(pose_to_vec(msg))
            elif topic == odom_topic:
                odom_times.append(ts)
                odom_positions.append(odom_to_vec(msg))

    if len(ref_times) == 0:
        raise RuntimeError(f"No reference messages found in {bag_path} on topic {REF_TOPIC}")
    if len(odom_times) == 0:
        raise RuntimeError(f"No odometry messages found in {bag_path} on topic {odom_topic}")

    ref_times = np.asarray(ref_times, dtype=float)
    ref_positions = np.asarray(ref_positions, dtype=float)
    odom_times = np.asarray(odom_times, dtype=float)
    odom_positions = np.asarray(odom_positions, dtype=float)

    return ref_times, ref_positions, odom_times, odom_positions


def interpolate_reference(ref_times, ref_positions, target_times):
    ref_interp = np.zeros((len(target_times), 3), dtype=float)
    for i in range(3):
        ref_interp[:, i] = np.interp(
            target_times,
            ref_times,
            ref_positions[:, i]
        )
    return ref_interp


def compute_error_metrics(ref_times, ref_positions, odom_times, odom_positions):
    ref_interp = interpolate_reference(ref_times, ref_positions, odom_times)

    err_xyz = odom_positions - ref_interp
    err_norm = np.linalg.norm(err_xyz, axis=1)

    metrics = {
        "num_samples": len(err_norm),
        "rms": float(np.sqrt(np.mean(err_norm ** 2))),
        "mean": float(np.mean(err_norm)),
        "max": float(np.max(err_norm)),
        "std": float(np.std(err_norm)),
        "final": float(err_norm[-1]),
        "mean_x": float(np.mean(np.abs(err_xyz[:, 0]))),
        "mean_y": float(np.mean(np.abs(err_xyz[:, 1]))),
        "mean_z": float(np.mean(np.abs(err_xyz[:, 2]))),
        "max_x": float(np.max(np.abs(err_xyz[:, 0]))),
        "max_y": float(np.max(np.abs(err_xyz[:, 1]))),
        "max_z": float(np.max(np.abs(err_xyz[:, 2]))),
    }

    return metrics, odom_times, err_xyz, err_norm


def settling_time(times, err_norm, threshold=0.10):
    """
    Returns the first time after which the error stays below threshold.
    If it never settles, returns None.
    """
    below = err_norm <= threshold
    for i in range(len(below)):
        if np.all(below[i:]):
            return float(times[i] - times[0])
    return None


def overshoot(err_norm, reference_step_mag):
    if reference_step_mag <= 1e-9:
        return None
    peak = float(np.max(err_norm))
    return 100.0 * peak / reference_step_mag


def estimate_reference_step(ref_positions):
    """
    Rough estimate of step magnitude using first and last reference samples.
    Useful mainly for hover/step commands.
    """
    return float(np.linalg.norm(ref_positions[-1] - ref_positions[0]))


def analyze_bag(bag_path, odom_topic):
    ref_times, ref_positions, odom_times, odom_positions = read_bag_positions(bag_path, odom_topic)
    metrics, times, err_xyz, err_norm = compute_error_metrics(
        ref_times, ref_positions, odom_times, odom_positions
    )

    step_mag = estimate_reference_step(ref_positions)
    metrics["reference_step_mag"] = step_mag

    st_10 = settling_time(times, err_norm, threshold=0.10)
    st_05 = settling_time(times, err_norm, threshold=0.05)

    metrics["settling_time_10cm"] = st_10
    metrics["settling_time_5cm"] = st_05
    metrics["overshoot_percent_like"] = overshoot(err_norm, step_mag)

    return metrics


def fmt_time(val):
    return "not settled" if val is None else f"{val:.3f} s"


def print_metrics(title, m):
    print(title)
    print(f"  Samples:              {m['num_samples']}")
    print(f"  RMS error:            {m['rms']:.4f} m")
    print(f"  Mean error:           {m['mean']:.4f} m")
    print(f"  Std error:            {m['std']:.4f} m")
    print(f"  Max error:            {m['max']:.4f} m")
    print(f"  Final error:          {m['final']:.4f} m")
    print(f"  Mean |ex|:            {m['mean_x']:.4f} m")
    print(f"  Mean |ey|:            {m['mean_y']:.4f} m")
    print(f"  Mean |ez|:            {m['mean_z']:.4f} m")
    print(f"  Max |ex|:             {m['max_x']:.4f} m")
    print(f"  Max |ey|:             {m['max_y']:.4f} m")
    print(f"  Max |ez|:             {m['max_z']:.4f} m")
    print(f"  Settling time <10cm:  {fmt_time(m['settling_time_10cm'])}")
    print(f"  Settling time <5cm:   {fmt_time(m['settling_time_5cm'])}")
    if m["overshoot_percent_like"] is None:
        print("  Overshoot-like:       n/a")
    else:
        print(f"  Overshoot-like:       {m['overshoot_percent_like']:.2f} %")
    print()


def print_comparison(raw_m, ekf_m):
    print("Comparison (EKF relative to RAW)")
    for key, label, unit in [
        ("rms", "RMS error", "m"),
        ("mean", "Mean error", "m"),
        ("max", "Max error", "m"),
        ("final", "Final error", "m"),
        ("mean_x", "Mean |ex|", "m"),
        ("mean_y", "Mean |ey|", "m"),
        ("mean_z", "Mean |ez|", "m"),
    ]:
        raw_val = raw_m[key]
        ekf_val = ekf_m[key]
        diff = ekf_val - raw_val
        pct = 100.0 * diff / raw_val if abs(raw_val) > 1e-12 else float("nan")
        print(f"  {label:18s}: {raw_val:.4f} -> {ekf_val:.4f} {unit}   "
              f"({diff:+.4f} {unit}, {pct:+.2f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare closed-loop tracking error from raw.bag and ekf.bag"
    )
    parser.add_argument("raw_bag", help="Bag recorded with raw odometry feedback loop")
    parser.add_argument("ekf_bag", help="Bag recorded with EKF odometry feedback loop")
    args = parser.parse_args()

    try:
        raw_metrics = analyze_bag(args.raw_bag, RAW_ODOM_TOPIC)
        ekf_metrics = analyze_bag(args.ekf_bag, EKF_ODOM_TOPIC)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n========== CLOSED-LOOP TRACKING COMPARISON ==========\n")
    print_metrics("RAW FEEDBACK LOOP", raw_metrics)
    print_metrics("EKF FEEDBACK LOOP", ekf_metrics)
    print_comparison(raw_metrics, ekf_metrics)
    print("=====================================================\n")


if __name__ == "__main__":
    main()
