#!/usr/bin/env python3
import rosbag
import numpy as np

BAG_FILE = "ekf_eval.bag"

TRUTH = "/hummingbird/ground_truth/odometry"
NOISY = "/hummingbird/odometry_sensor1/odometry"
EKF   = "/hummingbird/ekf/odometry"  # change if you evaluate _100hz

MAX_DT = 0.03  # seconds: reject matches farther than this (set None to disable)

def read_positions(bag, topic):
    data = []
    for _, msg, _t in bag.read_messages(topics=[topic]):
        # Prefer message header timestamp (sim time) over bag time
        ts = msg.header.stamp.to_sec()
        p = np.array([msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      msg.pose.pose.position.z], dtype=float)
        data.append((ts, p))
    # Ensure sorted by time
    data.sort(key=lambda x: x[0])
    return data

def nearest_neighbor_errors(truth_data, est_data, max_dt=None):
    """
    For each truth timestamp, pick the est sample with closest timestamp.
    Optionally reject pairs with |dt| > max_dt.
    """
    if len(truth_data) == 0 or len(est_data) == 0:
        return None

    t_est = np.array([te for te, _ in est_data], dtype=float)
    p_est = np.array([pe for _, pe in est_data], dtype=float)

    errors = []
    used = 0
    dropped = 0

    for t_truth, p_truth in truth_data:
        j = int(np.searchsorted(t_est, t_truth))

        candidates = []
        if j > 0:
            candidates.append(j - 1)
        if j < len(t_est):
            candidates.append(j)

        # Choose closest candidate
        j_best = min(candidates, key=lambda k: abs(t_est[k] - t_truth))
        dt = abs(t_est[j_best] - t_truth)

        if max_dt is not None and dt > max_dt:
            dropped += 1
            continue

        errors.append(p_est[j_best] - p_truth)
        used += 1

    errors = np.array(errors) if used > 0 else None
    return errors, used, dropped

def summarize_errors(errors):
    mean_xyz = np.mean(errors, axis=0)
    std_xyz  = np.std(errors, axis=0)
    rmse_xyz = np.sqrt(np.mean(errors**2, axis=0))

    norms = np.linalg.norm(errors, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm  = float(np.std(norms))
    rmse_norm = float(np.sqrt(np.mean(norms**2)))

    return mean_xyz, std_xyz, rmse_xyz, mean_norm, std_norm, rmse_norm

def print_block(name, errors, used, dropped):
    mean_xyz, std_xyz, rmse_xyz, mean_n, std_n, rmse_n = summarize_errors(errors)
    print(f"\n===== {name} =====")
    print(f"Pairs used: {used}, dropped (dt>{MAX_DT}s): {dropped}")
    print(f"Mean Error [x y z] (m): {mean_xyz}")
    print(f"Std Dev    [x y z] (m): {std_xyz}")
    print(f"RMSE       [x y z] (m): {rmse_xyz}")
    print(f"Mean |e| (m): {mean_n:.6f}")
    print(f"Std  |e| (m): {std_n:.6f}")
    print(f"RMSE |e| (m): {rmse_n:.6f}")

def main():
    print(f"Reading bag: {BAG_FILE}")
    with rosbag.Bag(BAG_FILE) as bag:
        truth = read_positions(bag, TRUTH)
        noisy = read_positions(bag, NOISY)
        ekf   = read_positions(bag, EKF)

    print(f"Samples: truth={len(truth)}, noisy={len(noisy)}, ekf={len(ekf)}")

    noisy_res = nearest_neighbor_errors(truth, noisy, max_dt=MAX_DT)
    ekf_res   = nearest_neighbor_errors(truth, ekf,   max_dt=MAX_DT)

    if noisy_res is None:
        print("ERROR: missing truth or noisy topic data.")
        return
    if ekf_res is None:
        print("ERROR: missing truth or ekf topic data. Is EKF running and recorded?")
        return

    noisy_err, used_n, dropped_n = noisy_res
    ekf_err,   used_e, dropped_e = ekf_res

    if noisy_err is None or len(noisy_err) == 0:
        print("ERROR: no valid noisy matches (check MAX_DT).")
        return
    if ekf_err is None or len(ekf_err) == 0:
        print("ERROR: no valid ekf matches (check MAX_DT).")
        return

    print_block("NOISY ODOMETRY vs TRUTH", noisy_err, used_n, dropped_n)
    print_block("EKF ESTIMATE vs TRUTH", ekf_err, used_e, dropped_e)

    # Improvement summary (norm RMSE)
    _, _, _, _, _, rmse_noisy = summarize_errors(noisy_err)
    _, _, _, _, _, rmse_ekf   = summarize_errors(ekf_err)
    improvement = (rmse_noisy - rmse_ekf) / rmse_noisy * 100.0 if rmse_noisy > 0 else 0.0

    print("\n===== IMPROVEMENT (norm RMSE) =====")
    print(f"Noisy RMSE |e|: {rmse_noisy:.6f} m")
    print(f"EKF   RMSE |e|: {rmse_ekf:.6f} m")
    print(f"Improvement: {improvement:.2f} %")

if __name__ == "__main__":
    main()
