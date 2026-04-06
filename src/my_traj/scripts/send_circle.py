#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from math import sin, cos, pi, atan2


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, sin(yaw / 2.0), cos(yaw / 2.0)


def make_pose(stamp, x, y, z, yaw):
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = "world"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def main():
    rospy.init_node("send_circle_slow")

    cmd_topic = rospy.get_param("~cmd_topic", "/hummingbird/command/pose")
    ref_topic = rospy.get_param("~ref_topic", "/hummingbird/reference_pose")
    publish_hz = float(rospy.get_param("~publish_hz", 30.0))

    center_x = float(rospy.get_param("~center_x", 0.0))
    center_y = float(rospy.get_param("~center_y", 0.0))
    center_z = float(rospy.get_param("~center_z", 1.0))
    radius = float(rospy.get_param("~radius", 1.5))
    period = float(rospy.get_param("~period", 25.0))

    face_forward = bool(rospy.get_param("~face_forward", False))
    fixed_yaw = float(rospy.get_param("~fixed_yaw", 0.0))

    cmd_pub = rospy.Publisher(cmd_topic, PoseStamped, queue_size=10)
    ref_pub = rospy.Publisher(ref_topic, PoseStamped, queue_size=10)

    rospy.loginfo("Publishing slow circular trajectory")
    rospy.loginfo(f"  cmd_topic:    {cmd_topic}")
    rospy.loginfo(f"  ref_topic:    {ref_topic}")
    rospy.loginfo(f"  center:       ({center_x}, {center_y}, {center_z})")
    rospy.loginfo(f"  radius:       {radius}")
    rospy.loginfo(f"  period:       {period}")
    rospy.loginfo(f"  publish_hz:   {publish_hz}")
    rospy.loginfo(f"  face_forward: {face_forward}")

    rate = rospy.Rate(publish_hz)
    t0 = rospy.Time.now().to_sec()
    omega = 2.0 * pi / period

    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0
        theta = omega * t

        x = center_x + radius * cos(theta)
        y = center_y + radius * sin(theta)
        z = center_z

        if face_forward:
            vx = -radius * omega * sin(theta)
            vy =  radius * omega * cos(theta)
            yaw = atan2(vy, vx)
        else:
            yaw = fixed_yaw

        now = rospy.Time.now()
        msg = make_pose(now, x, y, z, yaw)

        cmd_pub.publish(msg)
        ref_pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
