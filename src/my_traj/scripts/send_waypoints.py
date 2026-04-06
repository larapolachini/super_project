#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from math import sin, cos, pi, atan2

# =========================================================
# Helix trajectory parameters
# =========================================================

CENTER_X = 0.0
CENTER_Y = 0.0

RADIUS = 2.0              # helix radius [m]
Z0 = 1.0                  # initial altitude [m]
Z_SPEED = 0.05            # vertical speed [m/s]
Z_MAX = 3.0               # maximum altitude [m]

PERIOD = 20.0             # time for one full turn [s]
PUBLISH_HZ = 30           # publish rate [Hz]

FACE_FORWARD = False      # True -> yaw follows tangent
FIXED_YAW = 0.0           # used if FACE_FORWARD=False


def yaw_to_quaternion(yaw):
    qx = 0.0
    qy = 0.0
    qz = sin(yaw / 2.0)
    qw = cos(yaw / 2.0)
    return qx, qy, qz, qw


def main():
    rospy.init_node("send_helix_pose")
    pub = rospy.Publisher("/hummingbird/command/pose", PoseStamped, queue_size=10)
    rate = rospy.Rate(PUBLISH_HZ)

    rospy.loginfo("Publishing helix trajectory to /hummingbird/command/pose")

    t0 = rospy.Time.now().to_sec()
    omega = 2.0 * pi / PERIOD   # angular speed [rad/s]

    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0
        theta = omega * t

        # Helix position
        x = CENTER_X + RADIUS * cos(theta)
        y = CENTER_Y + RADIUS * sin(theta)
        z = Z0 + 0.8 * sin(0.2 *t)

        if z > Z_MAX:
            z = Z_MAX

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        if FACE_FORWARD:
            # Tangent direction of circular motion
            dx = -RADIUS * omega * sin(theta)
            dy =  RADIUS * omega * cos(theta)
            yaw = atan2(dy, dx)
        else:
            yaw = FIXED_YAW

        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
