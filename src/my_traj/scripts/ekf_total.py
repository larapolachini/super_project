#!/usr/bin/env python3
import rospy
import numpy as np

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_matrix


def quat_normalize(q):
    q = np.array(q, dtype=float).reshape(4,)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quat_to_rot(q):
    q = quat_normalize(q)
    return quaternion_matrix([q[0], q[1], q[2], q[3]])[:3, :3]


def omega_matrix(wx, wy, wz):
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx,   0.0,  wz, -wy],
        [wy,  -wz,  0.0,  wx],
        [wz,   wy, -wx,  0.0]
    ], dtype=float)


class EKFPosAttImuOdom:
    """
    13-state EKF-like filter:
      x = [p(3), v(3), q(4), w(3)]^T
        = [px,py,pz, vx,vy,vz, qx,qy,qz,qw, wx,wy,wz]^T

    Predict from IMU:
      - accel -> world translational prediction
      - gyro  -> quaternion propagation
      - angular velocity state follows constant model

    Update from odometry:
      z_odom = [p_odom, v_odom]

    Update from IMU measurement:
      z_imu = [q_imu, w_imu]

    Notes:
      - Quaternion is renormalized after predict and update.
      - This is a practical EKF-style extension, not a full error-state attitude EKF.
      - IMU orientation is treated as a measurement for attitude correction.
    """

    def __init__(self):
        # Topics
        self.imu_topic  = rospy.get_param("~imu_topic",  "/hummingbird/imu")
        self.odom_topic = rospy.get_param("~odom_topic", "/hummingbird/odometry_sensor1/odometry")
        self.out_topic  = rospy.get_param("~out_topic",  "/hummingbird/ekf/odometry")

        # Gravity in ENU world
        self.g = np.array([0.0, 0.0, -9.81], dtype=float)

        # Noise params
        self.accel_noise_std = float(rospy.get_param("~accel_noise_std", 20.0))
        self.gyro_noise_std  = float(rospy.get_param("~gyro_noise_std", 1.0))      # rad/s process
        self.pos_meas_std    = float(rospy.get_param("~pos_meas_std", 0.15))
        self.vel_meas_std    = float(rospy.get_param("~vel_meas_std", 0.30))
        self.att_meas_std    = float(rospy.get_param("~att_meas_std", 0.05))       # quaternion pseudo std
        self.omega_meas_std  = float(rospy.get_param("~omega_meas_std", 0.10))     # rad/s

        # IMU handling
        self.max_dt = float(rospy.get_param("~max_dt", 0.05))
        self.max_accel_norm = float(rospy.get_param("~max_accel_norm", 30.0))
        self.max_gyro_norm  = float(rospy.get_param("~max_gyro_norm", 20.0))

        self.use_R_transpose = bool(rospy.get_param("~use_R_transpose", True))
        self.use_gravity_add = bool(rospy.get_param("~use_gravity_add", True))

        self.P_floor = float(rospy.get_param("~P_floor", 1e-6))

        # State: [p(3), v(3), q(4), w(3)]
        self.nx = 13
        self.x = np.zeros((self.nx, 1), dtype=float)
        self.x[9, 0] = 1.0  # qw = 1
        self.P = np.eye(self.nx, dtype=float) * 1.0

        self.initialized = False
        self.last_imu_time = None

        # Cache latest raw IMU orientation
        self.latest_q_meas = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        # Measurement covariances
        self.R_odom = np.diag([
            self.pos_meas_std**2, self.pos_meas_std**2, self.pos_meas_std**2,
            self.vel_meas_std**2, self.vel_meas_std**2, self.vel_meas_std**2
        ])

        self.R_imu = np.diag([
            self.att_meas_std**2, self.att_meas_std**2, self.att_meas_std**2, self.att_meas_std**2,
            self.omega_meas_std**2, self.omega_meas_std**2, self.omega_meas_std**2
        ])

        self.pub = rospy.Publisher(self.out_topic, Odometry, queue_size=10)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_cb, queue_size=200)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=50)

        rospy.loginfo("EKF 13-state started:")
        rospy.loginfo(f"  IMU topic:  {self.imu_topic}")
        rospy.loginfo(f"  Odom topic: {self.odom_topic}")
        rospy.loginfo(f"  Out topic:  {self.out_topic}")

    def imu_cb(self, msg: Imu):
        q_meas = quat_normalize([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        self.latest_q_meas = q_meas

        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=float)

        accel_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=float)

        if not self.initialized:
            return

        t = msg.header.stamp.to_sec()
        if self.last_imu_time is None:
            self.last_imu_time = t
            return

        dt = t - self.last_imu_time
        self.last_imu_time = t

        if dt <= 0.0 or dt > self.max_dt:
            return

        if np.linalg.norm(accel_body) > self.max_accel_norm:
            return
        if np.linalg.norm(gyro) > self.max_gyro_norm:
            return

        # Current state pieces
        p = self.x[0:3, 0].copy()
        v = self.x[3:6, 0].copy()
        q = quat_normalize(self.x[6:10, 0].copy())
        w = self.x[10:13, 0].copy()

        # For prediction, use measured gyro directly as current body rates
        w_pred_input = gyro.copy()

        # Rotate accel to world using current attitude estimate
        R_wb = quat_to_rot(q)
        if self.use_R_transpose:
            f_world = R_wb.T.dot(accel_body)
        else:
            f_world = R_wb.dot(accel_body)

        if self.use_gravity_add:
            a_world = f_world + self.g
        else:
            a_world = f_world - self.g

        # ---- Nonlinear state prediction ----
        p_new = p + v * dt + 0.5 * a_world * dt * dt
        v_new = v + a_world * dt

        Omega = omega_matrix(w_pred_input[0], w_pred_input[1], w_pred_input[2])
        q_dot = 0.5 * Omega.dot(q.reshape(4, 1))
        q_new = q.reshape(4, 1) + q_dot * dt
        q_new = quat_normalize(q_new.flatten())

        # Angular velocity state: constant model driven by gyro measurement
        w_new = w_pred_input.copy()

        self.x[0:3, 0] = p_new
        self.x[3:6, 0] = v_new
        self.x[6:10, 0] = q_new
        self.x[10:13, 0] = w_new

        # ---- Approximate covariance prediction ----
        F = np.eye(self.nx, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # crude coupling from angular velocity into quaternion
        # q_k+1 ≈ q_k + 0.5*Omega(w)*q*dt
        # We keep this simple for practical use.
        F[6:10, 10:13] = 0.5 * dt * np.array([
            [ self.x[9,0], -self.x[8,0],  self.x[7,0]],
            [ self.x[8,0],  self.x[9,0], -self.x[6,0]],
            [-self.x[7,0],  self.x[6,0],  self.x[9,0]],
            [-self.x[6,0], -self.x[7,0], -self.x[8,0]]
        ], dtype=float)

        Q = np.zeros((self.nx, self.nx), dtype=float)

        # accel process noise -> p,v
        Ba = np.zeros((6, 3), dtype=float)
        Ba[0:3, :] = 0.5 * dt * dt * np.eye(3)
        Ba[3:6, :] = dt * np.eye(3)
        Qa = (self.accel_noise_std ** 2) * (Ba.dot(Ba.T))
        Q[0:6, 0:6] += Qa

        # gyro process noise -> q,w
        Q[6:10, 6:10] += np.eye(4) * (self.gyro_noise_std ** 2) * dt * dt
        Q[10:13, 10:13] += np.eye(3) * (self.gyro_noise_std ** 2) * dt

        self.P = F.dot(self.P).dot(F.T) + Q
        self.P = 0.5 * (self.P + self.P.T) + np.eye(self.nx) * self.P_floor

        # ---- IMU measurement update for q and w ----
        z = np.zeros((7, 1), dtype=float)
        z[0:4, 0] = q_meas
        z[4:7, 0] = gyro

        H = np.zeros((7, self.nx), dtype=float)
        H[0:4, 6:10] = np.eye(4)
        H[4:7, 10:13] = np.eye(3)

        # Quaternion sign consistency:
        q_est = quat_normalize(self.x[6:10, 0])
        if np.dot(q_est, q_meas) < 0.0:
            z[0:4, 0] *= -1.0

        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + self.R_imu
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        self.x[6:10, 0] = quat_normalize(self.x[6:10, 0])

        I = np.eye(self.nx)
        self.P = (I - K.dot(H)).dot(self.P)
        self.P = 0.5 * (self.P + self.P.T) + np.eye(self.nx) * self.P_floor

    def odom_cb(self, msg: Odometry):
        if not self.initialized:
            self.x[0, 0] = msg.pose.pose.position.x
            self.x[1, 0] = msg.pose.pose.position.y
            self.x[2, 0] = msg.pose.pose.position.z

            self.x[3, 0] = msg.twist.twist.linear.x
            self.x[4, 0] = msg.twist.twist.linear.y
            self.x[5, 0] = msg.twist.twist.linear.z

            # initialize quaternion from odom if available, else IMU cache
            q0 = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ], dtype=float)

            if np.linalg.norm(q0) < 1e-6:
                q0 = self.latest_q_meas.copy()

            q0 = quat_normalize(q0)
            self.x[6:10, 0] = q0
            self.x[10:13, 0] = 0.0

            self.P = np.eye(self.nx, dtype=float) * 0.5
            self.last_imu_time = None
            self.initialized = True

            rospy.loginfo("EKF initialized from first odometry message.")
            self.publish_estimate(msg.header.stamp)
            return

        z = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
            [msg.pose.pose.position.z],
            [msg.twist.twist.linear.x],
            [msg.twist.twist.linear.y],
            [msg.twist.twist.linear.z]
        ], dtype=float)

        H = np.zeros((6, self.nx), dtype=float)
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + self.R_odom
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        self.x[6:10, 0] = quat_normalize(self.x[6:10, 0])

        I = np.eye(self.nx)
        self.P = (I - K.dot(H)).dot(self.P)
        self.P = 0.5 * (self.P + self.P.T) + np.eye(self.nx) * self.P_floor

        self.publish_estimate(msg.header.stamp)

    def publish_estimate(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "hummingbird/base_link"

        # Position
        odom.pose.pose.position.x = float(self.x[0, 0])
        odom.pose.pose.position.y = float(self.x[1, 0])
        odom.pose.pose.position.z = float(self.x[2, 0])

        # Attitude quaternion
        q = quat_normalize(self.x[6:10, 0])
        odom.pose.pose.orientation.x = float(q[0])
        odom.pose.pose.orientation.y = float(q[1])
        odom.pose.pose.orientation.z = float(q[2])
        odom.pose.pose.orientation.w = float(q[3])

        # Linear velocity
        odom.twist.twist.linear.x = float(self.x[3, 0])
        odom.twist.twist.linear.y = float(self.x[4, 0])
        odom.twist.twist.linear.z = float(self.x[5, 0])

        # Angular velocity
        odom.twist.twist.angular.x = float(self.x[10, 0])
        odom.twist.twist.angular.y = float(self.x[11, 0])
        odom.twist.twist.angular.z = float(self.x[12, 0])

        # Covariances
        odom.pose.covariance[0]  = float(self.P[0, 0])
        odom.pose.covariance[7]  = float(self.P[1, 1])
        odom.pose.covariance[14] = float(self.P[2, 2])

        # orientation covariance slots in 6x6 pose covariance:
        odom.pose.covariance[21] = float(self.P[6, 6])   # roll-ish proxy
        odom.pose.covariance[28] = float(self.P[7, 7])   # pitch-ish proxy
        odom.pose.covariance[35] = float(self.P[8, 8])   # yaw-ish proxy

        odom.twist.covariance[0]  = float(self.P[3, 3])
        odom.twist.covariance[7]  = float(self.P[4, 4])
        odom.twist.covariance[14] = float(self.P[5, 5])
        odom.twist.covariance[21] = float(self.P[10, 10])
        odom.twist.covariance[28] = float(self.P[11, 11])
        odom.twist.covariance[35] = float(self.P[12, 12])

        self.pub.publish(odom)


def main():
    rospy.init_node("ekf_pos_att_imu_odom")
    EKFPosAttImuOdom()
    rospy.spin()


if __name__ == "__main__":
    main()
