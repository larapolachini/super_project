#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from math import sin, cos


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
    rospy.init_node("send_step")

    cmd_topic = rospy.get_param("~cmd_topic", "/hummingbird/command/pose")
    ref_topic = rospy.get_param("~ref_topic", "/hummingbird/reference_pose")
    publish_hz = float(rospy.get_param("~publish_hz", 30.0))
    hold_time = float(rospy.get_param("~hold_time", 6.0))
    fixed_yaw = float(rospy.get_param("~fixed_yaw", 0.0))

    step_points = rospy.get_param(
        "~step_points",
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.2],
            [0.0, 0.0, 1.0],
        ],
    )

    cmd_pub = rospy.Publisher(cmd_topic, PoseStamped, queue_size=10)
    ref_pub = rospy.Publisher(ref_topic, PoseStamped, queue_size=10)

    rospy.loginfo("Publishing step-and-hold trajectory")
    rospy.loginfo(f"  cmd_topic:  {cmd_topic}")
    rospy.loginfo(f"  ref_topic:  {ref_topic}")
    rospy.loginfo(f"  hold_time:  {hold_time}")
    rospy.loginfo(f"  publish_hz: {publish_hz}")

    rate = rospy.Rate(publish_hz)
    t0 = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0
        idx = int(t // hold_time) % len(step_points)
        x, y, z = step_points[idx]

        now = rospy.Time.now()
        msg = make_pose(now, x, y, z, fixed_yaw)

        cmd_pub.publish(msg)
        ref_pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
