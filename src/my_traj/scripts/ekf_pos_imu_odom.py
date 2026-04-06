#!/usr/bin/env python3
import rospy
import numpy as np

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_matrix


class EKFPosImuOdom:
    """
    6-state KF/EKF for translation:
      x = [p_w; v_w] = [px,py,pz,vx,vy,vz]^T

    Predict (IMU):
      p <- p + v*dt + 0.5*a_w*dt^2
      v <- v + a_w*dt

    Update (odometry):
      z = [p_odom; v_odom], H = I

    Notes:
      - We publish ONLY on odom updates to keep output rate stable (~odom rate).
      - IMU orientation is used only to rotate accel; attitude is not estimated.
      - RotorS IMU conventions can differ; we expose use_R_transpose to flip rotation.
    """

    def __init__(self):
        # Topics
        self.imu_topic  = rospy.get_param("~imu_topic",  "/hummingbird/imu")
        self.odom_topic = rospy.get_param("~odom_topic", "/hummingbird/odometry_sensor1/odometry")
        self.out_topic  = rospy.get_param("~out_topic",  "/hummingbird/ekf/odometry")

        # World gravity (ENU z-up => gravity is negative z acceleration)
        # If your results look inverted, flip this sign OR set use_gravity_add=False.
        self.g = np.array([0.0, 0.0, -9.81], dtype=float)

        # Tuning params
        # With your increased odom noise, these defaults match your sensor config.
        self.accel_noise_std = float(rospy.get_param("~accel_noise_std", 20.0))  # m/s^2
        self.pos_meas_std    = float(rospy.get_param("~pos_meas_std", 0.15))     # m
        self.vel_meas_std    = float(rospy.get_param("~vel_meas_std", 0.30))     # m/s

        # IMU handling
        self.max_dt = float(rospy.get_param("~max_dt", 0.05))
        self.max_accel_norm = float(rospy.get_param("~max_accel_norm", 30.0))

        # If True: f_world = R^T f_body (often the correct convention)
        # If False: f_world = R f_body
        self.use_R_transpose = bool(rospy.get_param("~use_R_transpose", True))

        # If True: a_world = f_world + g (specific force includes gravity)
        # If False: a_world = f_world - g
        self.use_gravity_add = bool(rospy.get_param("~use_gravity_add", True))

        # Numerical stability
        self.P_floor = float(rospy.get_param("~P_floor", 1e-6))  # covariance floor

        # Process noise variance
        self.sigma_a2 = self.accel_noise_std ** 2

        # Measurement noise
        self.R6 = np.diag([
            self.pos_meas_std**2, self.pos_meas_std**2, self.pos_meas_std**2,
            self.vel_meas_std**2, self.vel_meas_std**2, self.vel_meas_std**2
        ])

        # State and covariance
        self.x = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6, dtype=float) * 1.0

        self.initialized = False

        # IMU time + orientation cache
        self.last_imu_time = None
        self.latest_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        # Subscribers/publisher
        self.pub = rospy.Publisher(self.out_topic, Odometry, queue_size=10)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_cb, queue_size=200)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=50)

        rospy.loginfo("EKF (6-state) started:")
        rospy.loginfo(f"  IMU topic:  {self.imu_topic}")
        rospy.loginfo(f"  Odom topic: {self.odom_topic}")
        rospy.loginfo(f"  Out topic:  {self.out_topic}")
        rospy.loginfo(f"  accel_noise_std={self.accel_noise_std}, pos_meas_std={self.pos_meas_std}, vel_meas_std={self.vel_meas_std}")
        rospy.loginfo(f"  use_R_transpose={self.use_R_transpose}, use_gravity_add={self.use_gravity_add}")
        rospy.loginfo(f"  max_dt={self.max_dt}, max_accel_norm={self.max_accel_norm}")

    def imu_cb(self, msg: Imu):
        # Cache quaternion always (for visualization)
        q = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=float)
        self.latest_q = q

        # Do not predict until initialized from odom
        if not self.initialized:
            return

        t = msg.header.stamp.to_sec()
        if self.last_imu_time is None:
            self.last_imu_time = t
            return

        dt = t - self.last_imu_time
        self.last_imu_time = t

        # Clamp dt for stability
        if dt <= 0.0 or dt > self.max_dt:
            return

        # IMU specific force in body frame (as published by RotorS)
        f_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=float)

        # Reject spikes
        if np.linalg.norm(f_body) > self.max_accel_norm:
            return

        # Rotate to world frame
        R_wb = quaternion_matrix([q[0], q[1], q[2], q[3]])[:3, :3]
        if self.use_R_transpose:
            f_world = R_wb.T.dot(f_body)
        else:
            f_world = R_wb.dot(f_body)

        # Convert specific force to acceleration
        if self.use_gravity_add:
            a_world = f_world + self.g
        else:
            a_world = f_world - self.g

        # Discrete-time model matrices
        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        B = np.zeros((6, 3), dtype=float)
        B[0:3, :] = 0.5 * (dt ** 2) * np.eye(3)
        B[3:6, :] = dt * np.eye(3)

        u = a_world.reshape((3, 1))

        # Predict
        self.x = F.dot(self.x) + B.dot(u)

        Q = self.sigma_a2 * (B.dot(B.T))
        self.P = F.dot(self.P).dot(F.T) + Q

        # Covariance floor
        self.P = self.P + np.eye(6) * self.P_floor

    def odom_cb(self, msg: Odometry):
        # Initialize from first odometry message
        if not self.initialized:
            self.x[0, 0] = msg.pose.pose.position.x
            self.x[1, 0] = msg.pose.pose.position.y
            self.x[2, 0] = msg.pose.pose.position.z
            self.x[3, 0] = msg.twist.twist.linear.x
            self.x[4, 0] = msg.twist.twist.linear.y
            self.x[5, 0] = msg.twist.twist.linear.z

            self.P = np.eye(6, dtype=float) * 0.5  # initial uncertainty
            self.last_imu_time = None
            self.initialized = True
            rospy.loginfo("EKF initialized from first odometry message.")
            self.publish_estimate(msg.header.stamp)
            return

        # Measurement
        z = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
            [msg.pose.pose.position.z],
            [msg.twist.twist.linear.x],
            [msg.twist.twist.linear.y],
            [msg.twist.twist.linear.z]
        ], dtype=float)

        H = np.eye(6, dtype=float)

        # Innovation
        y = z - H.dot(self.x)

        S = H.dot(self.P).dot(H.T) + self.R6
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        # Update
        self.x = self.x + K.dot(y)
        self.P = (np.eye(6) - K.dot(H)).dot(self.P)

        # Symmetrize + floor
        self.P = 0.5 * (self.P + self.P.T) + np.eye(6) * self.P_floor

        # Publish ONLY here (stable output rate)
        self.publish_estimate(msg.header.stamp)

    def publish_estimate(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "hummingbird/base_link"

        odom.pose.pose.position.x = float(self.x[0, 0])
        odom.pose.pose.position.y = float(self.x[1, 0])
        odom.pose.pose.position.z = float(self.x[2, 0])

        # publish IMU orientation for visualization
        odom.pose.pose.orientation.x = float(self.latest_q[0])
        odom.pose.pose.orientation.y = float(self.latest_q[1])
        odom.pose.pose.orientation.z = float(self.latest_q[2])
        odom.pose.pose.orientation.w = float(self.latest_q[3])

        odom.twist.twist.linear.x = float(self.x[3, 0])
        odom.twist.twist.linear.y = float(self.x[4, 0])
        odom.twist.twist.linear.z = float(self.x[5, 0])

        # Some covariance entries (optional)
        odom.pose.covariance[0]  = float(self.P[0, 0])
        odom.pose.covariance[7]  = float(self.P[1, 1])
        odom.pose.covariance[14] = float(self.P[2, 2])
        odom.twist.covariance[0]  = float(self.P[3, 3])
        odom.twist.covariance[7]  = float(self.P[4, 4])
        odom.twist.covariance[14] = float(self.P[5, 5])

        self.pub.publish(odom)


def main():
    rospy.init_node("ekf_pos_imu_odom")
    EKFPosImuOdom()
    rospy.spin()


if __name__ == "__main__":
    main()
