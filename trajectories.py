#!/usr/bin/env python3
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


BAG_FILE = "ekf_eval.bag"

TRUTH_TOPIC = "/hummingbird/ground_truth/odometry"
NOISY_TOPIC = "/hummingbird/odometry_sensor1/odometry"
EKF_TOPIC   = "/hummingbird/ekf/odometry"


def read_xyz(bag_path: str, topic: str):
    t_list = []
    xyz_list = []

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            ts = msg.header.stamp.to_sec()
            p = msg.pose.pose.position
            t_list.append(ts)
            xyz_list.append([p.x, p.y, p.z])

    if not t_list:
        return np.array([]), np.zeros((0, 3))

    t = np.array(t_list, dtype=float)
    xyz = np.array(xyz_list, dtype=float)

    idx = np.argsort(t)
    return t[idx], xyz[idx]


def plot_3d_trajectories(truth_xyz, noisy_xyz, ekf_xyz):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(truth_xyz[:, 0], truth_xyz[:, 1], truth_xyz[:, 2], label="Ground Truth", linewidth=2)
    ax.plot(noisy_xyz[:, 0], noisy_xyz[:, 1], noisy_xyz[:, 2], label="Noisy Odometry", alpha=0.7)
    ax.plot(ekf_xyz[:, 0], ekf_xyz[:, 1], ekf_xyz[:, 2], label="EKF", linewidth=2)

    ax.set_title("3D Trajectories: Truth vs Noisy vs EKF")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Equal axis scaling
    all_xyz = np.vstack([truth_xyz, noisy_xyz, ekf_xyz])
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    spans = maxs - mins
    span = float(np.max(spans))
    centers = (maxs + mins) / 2.0

    ax.set_xlim(centers[0] - span/2, centers[0] + span/2)
    ax.set_ylim(centers[1] - span/2, centers[1] + span/2)
    ax.set_zlim(centers[2] - span/2, centers[2] + span/2)

    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # ✅ Save files
    plt.savefig("trajectories_3d.png", dpi=300)
    plt.savefig("trajectories_3d.pdf")

    print("Saved:")
    print(" - trajectories_3d.png")
    print(" - trajectories_3d.pdf")

    plt.show()


def main():
    _, truth_xyz = read_xyz(BAG_FILE, TRUTH_TOPIC)
    _, noisy_xyz = read_xyz(BAG_FILE, NOISY_TOPIC)
    _, ekf_xyz   = read_xyz(BAG_FILE, EKF_TOPIC)

    if truth_xyz.size == 0 or noisy_xyz.size == 0 or ekf_xyz.size == 0:
        raise RuntimeError("Missing one or more topics in bag file.")

    plot_3d_trajectories(truth_xyz, noisy_xyz, ekf_xyz)


if __name__ == "__main__":
    main()
