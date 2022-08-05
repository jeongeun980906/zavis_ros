#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from pynput import keyboard
import math
from geometry_msgs.msg import Twist
import time
pub=rospy.Publisher('/RosAria/cmd_vel', Twist, queue_size=10)
msg=Twist()
msg.linear.x=0
msg.linear.y=0
msg.linear.z=0
msg.angular.x=0
msg.angular.y=0
msg.angular.z=0
k=None

def on_press(key):
    k=key.char
    if k == "w":
        msg.linear.x=0.2
        msg.angular.z=0
        for _ in range(10):
            pub.publish(msg)
    elif k == "s":
        msg.linear.x=-0.2
        msg.angular.z=0
        for _ in range(10):
            pub.publish(msg)
    elif k == "a":
        msg.linear.x=0.05
        msg.angular.z=0.1*math.pi
        for _ in range(10):
            pub.publish(msg)
    elif k == "d":
        msg.linear.x=0.05
        msg.angular.z=-0.1*math.pi
        for _ in range(10):
            pub.publish(msg)
    elif k == "q":
        msg.linear.x=0.0
        msg.angular.z=0
        for _ in range(10):
            pub.publish(msg)

def on_release(key):
    # msg.linear.x=0
    # msg.linear.y=0
    # pub.publish(msg)
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def main():
    rospy.init_node('move_teltop', anonymous=True)
    pub.publish(msg)
    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
    rospy.spin()

if __name__ =="__main__":
    main()