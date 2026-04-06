#!/usr/bin/env python3
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

def extract_trajectory(bag_path, topic):
    times = []
    x, y, z = [], [], []

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            times.append(t.to_sec())
            x.append(msg.pose.pose.position.x)
            y.append(msg.pose.pose.position.y)
            z.append(msg.pose.pose.position.z)

    times = np.array(times)
    times = times - times[0]  # normalize time

    return times, np.array(x), np.array(y), np.array(z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_bag")
    parser.add_argument("ekf_bag")
    args = parser.parse_args()

    # Topics
    raw_topic = "/hummingbird/odometry_sensor1/odometry"
    ekf_topic = "/hummingbird/ekf/odometry"

    print("Loading raw trajectory...")
    t_raw, x_raw, y_raw, z_raw = extract_trajectory(args.raw_bag, raw_topic)

    print("Loading EKF trajectory...")
    t_ekf, x_ekf, y_ekf, z_ekf = extract_trajectory(args.ekf_bag, ekf_topic)

    # =========================
    # 1. XY trajectory
    # =========================
    plt.figure()
    plt.plot(x_raw, y_raw, label="Raw", linewidth=2)
    plt.plot(x_ekf, y_ekf, label="EKF", linewidth=2)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("XY Trajectory")
    plt.legend()
    plt.axis('equal')
    plt.grid()

    # =========================
    # 2. Position vs time
    # =========================
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(t_raw, x_raw, label="Raw")
    plt.plot(t_ekf, x_ekf, label="EKF")
    plt.ylabel("X [m]")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t_raw, y_raw, label="Raw")
    plt.plot(t_ekf, y_ekf, label="EKF")
    plt.ylabel("Y [m]")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t_raw, z_raw, label="Raw")
    plt.plot(t_ekf, z_ekf, label="EKF")
    plt.xlabel("Time [s]")
    plt.ylabel("Z [m]")
    plt.grid()

    plt.suptitle("Position vs Time")

    # =========================
    # 3. 3D trajectory
    # =========================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_raw, y_raw, z_raw, label="Raw")
    ax.plot(x_ekf, y_ekf, z_ekf, label="EKF")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D Trajectory")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
