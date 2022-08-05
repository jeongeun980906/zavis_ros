#!/usr/bin/env python

import sys
import math
import rospy
from nav_msgs.srv import GetPlan,GetPlanResponse
from geometry_msgs.msg import PoseStamped

rospy.wait_for_service('/move_base/NavfnROS/make_plan')
request = rospy.ServiceProxy('/move_base/NavfnROS/make_plan', GetPlan)
start = PoseStamped()
start.header.frame_id = 'map'
start.pose.orientation.w = 1
start.pose.position.x = 0

goal = PoseStamped()
goal.header.frame_id = 'map'
goal.pose.orientation.w = 1
goal.pose.position.x = 1

resp = request(start,goal,0.1)
waypoints = resp.plan.poses
res = 0
px = waypoints[0].pose.position.x
py = waypoints[0].pose.position.y
for wp in waypoints[1:]:
    x = wp.pose.position.x
    y = wp.pose.position.y
    res += math.sqrt((px-x)**2+(py-y)**2)
    px = x
    py = y
print(res)