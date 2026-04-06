#!/usr/bin/env python3
import rosbag
import numpy as np

bag_file = "odom_noise_test.bag"

truth_topic = "/hummingbird/ground_truth/odometry"
sensor_topic = "/hummingbird/odometry_sensor1/odometry"

# Store data
truth_data = []
sensor_data = []

print("Reading bag...")

with rosbag.Bag(bag_file) as bag:
    for topic, msg, t in bag.read_messages(topics=[truth_topic, sensor_topic]):
        if topic == truth_topic:
            truth_data.append((
                t.to_sec(),
                np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z
                ])
            ))
        elif topic == sensor_topic:
            sensor_data.append((
                t.to_sec(),
                np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z
                ])
            ))

truth_data = np.array(truth_data, dtype=object)
sensor_data = np.array(sensor_data, dtype=object)

print("Synchronizing...")

errors = []

j = 0
for t_truth, p_truth in truth_data:
    while j < len(sensor_data)-1 and sensor_data[j+1][0] < t_truth:
        j += 1

    t_sensor, p_sensor = sensor_data[j]
    errors.append(p_sensor - p_truth)

errors = np.array(errors)

mean_error = np.mean(errors, axis=0)
std_error = np.std(errors, axis=0)
rmse = np.sqrt(np.mean(errors**2, axis=0))

print("\n===== POSITION ERROR RESULTS =====")
print(f"Mean Error [x y z] (m): {mean_error}")
print(f"Std Dev    [x y z] (m): {std_error}")
print(f"RMSE       [x y z] (m): {rmse}")

