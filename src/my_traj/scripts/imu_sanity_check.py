#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Imu
from tf.transformations import quaternion_matrix

class IMUSanityCheck:

    def __init__(self):
        self.g = np.array([0.0, 0.0, -9.81])
        rospy.Subscriber("/hummingbird/imu", Imu, self.imu_cb)
        rospy.loginfo("IMU sanity check started...")

    def imu_cb(self, msg):

        # Acceleration in IMU/body frame
        f_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Quaternion from IMU message
        q = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

        # Rotation matrix
        R = quaternion_matrix(q)[:3, :3]

        # Candidate 1: assume quaternion is body->world
        a1 = R.dot(f_body) + self.g

        # Candidate 2: assume quaternion is world->body
        a2 = R.T.dot(f_body) + self.g

        print("\n--- IMU SANITY CHECK ---")
        print("Raw accel (body):", np.round(f_body, 3))
        print("Candidate 1 (R f + g):     ", np.round(a1, 3))
        print("Candidate 2 (R^T f + g):   ", np.round(a2, 3))

def main():
    rospy.init_node("imu_sanity_check")
    IMUSanityCheck()
    rospy.spin()

if __name__ == "__main__":
    main()
