#!/usr/bin/env python3

import queue
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped,PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
# from control_msgs.msg import JointControllerState
# from move_base_msgs.msg import MoveBaseActionResult
from actionlib_msgs.msg import GoalStatusArray,GoalID
from sensor_msgs.msg import JointState
from dynamixel_workbench_msgs.srv import DynamixelCommand
from nav_msgs.srv import GetPlan
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry

import math

def deg2rad(deg):
    return deg/180*math.pi

def rad2deg(rad):
    return rad/math.pi*180

def deg2command(angle):
    return int(1024*(angle/90) + 2048)

class ros_controller:
    def __init__(self):
        self.cam_pose_pub = rospy.Publisher("/pioneer3at/cam_position_controller/command",Float64,queue_size = 10)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal",PoseStamped,queue_size = 10)
        self.goal_cancel = rospy.Publisher("/move_base/cancel",GoalID,queue_size=10)
        
    def move2point(self,goal_pose, goal_rot = None,imshow_grid=None,scenemap=None,NUM_POINTS=0):
        '''
        args:
            goal_pose: dict (x,y,yaw)
        '''
        data = PoseStamped()
        data.header.frame_id = 'map'
        
        if goal_rot != None:
            flag = True
            goal_rot = deg2rad(goal_rot)
            q = quaternion_from_euler(0, 0, goal_rot)
        else:
            flag = False
            cpos = self.get_pose()
            crot = math.atan2(goal_pose['y']-cpos['y'],goal_pose['x']-cpos['x'])
            q = quaternion_from_euler(0,0,crot)
        data.pose.orientation.x = q[0]
        data.pose.orientation.y = q[1]
        data.pose.orientation.z = q[2]
        data.pose.orientation.w = q[3]
        data.pose.position.x = goal_pose['x']
        data.pose.position.y = goal_pose['y']
        data.pose.position.z = 0
        while True:
            self.goal_pub.publish(data)
            rospy.sleep(0.5)
            resp = rospy.wait_for_message("/move_base/status",GoalStatusArray)
            resp = resp.status_list
            if len(resp)>0:
                text = resp[-1].text
                # print(len(text))
                if text == 'This goal has been accepted by the simple action server':
                    break
            rospy.sleep(0.01)
        if flag:
            for _ in range(int(3e2)):
                try:
                    resp = rospy.wait_for_message("/move_base/status",GoalStatusArray,2)
                    # print(resp)
                except:
                    break
                cpos = self.get_pose()
                if cpos == None:
                    break
                resp = resp.status_list
                if len(resp)>0:
                    text = resp[-1].text
                    goal_id = resp[-1].goal_id.id
                    scenemap.color_path(None,imshow_grid,cpos,NUM_POINTS)

                    # print(goal_id)
                    if text == 'Goal reached.':
                        scenemap.color_path(cpos,imshow_grid,None,NUM_POINTS)
                        return True
            cancel = GoalID()
            cancel.id = goal_id
            for _ in range(10):
                self.goal_cancel.publish(cancel)
            scenemap.color_path(cpos,imshow_grid,None,NUM_POINTS)
            return False
        else:
            for _ in range(int(3e2)):
                cpos = self.get_pose()
                scenemap.color_path(None,imshow_grid,cpos,NUM_POINTS)
                dis = math.sqrt((cpos['x']-goal_pose['x'])**2+(cpos['y']-goal_pose['y'])**2)
                if dis<1e-2:
                    scenemap.color_path(cpos,imshow_grid,None,NUM_POINTS)
                    return True
            resp = rospy.wait_for_message("/move_base/status",GoalStatusArray)
            resp = resp.status_list
            goal_id = resp[-1].goal_id.id
            cancel = GoalID()
            cancel.id = goal_id
            for _ in range(10):
                self.goal_cancel.publish(cancel)
            scenemap.color_path(cpos,imshow_grid,None,NUM_POINTS)
            return False

    # def move_cam(self,angle):
    #     '''
    #     args:
    #         rotation angle: -180 to 180
    #     '''
    #     rad_ang = deg2rad(angle)
    #     data = Float64()
    #     data.data = rad_ang
    #     for _ in range(100):
    #         self.cam_pose_pub.publish(data)
    #         c_angle = self.get_cam_pose()
    #         if abs(c_angle-rad_ang)< 1e-2:
    #             return    
    #         else:
    #             rospy.sleep(0.1)
    #     return False
    def move_cam(self,angles):
        '''
        args:
            rotation angles: List 
                    ID 1: -180 to 180
                    ID 2: -45 to 45
        '''
        rospy.wait_for_service('/dynamixel_workbench/dynamixel_command')
        i =1
        res = True
        for angle in angles:
            if i == 2 :
                if angle>45 or angle<-45:
                    print("angle not in limit")
                    continue
            angle = deg2command(angle)
            request = rospy.ServiceProxy('/dynamixel_workbench/dynamixel_command', DynamixelCommand)
            resp = request("",i,"Goal_Position",angle)
            i +=1
            res *= resp.comm_result
        # print(res)
        rospy.sleep(0.5)
        return res

    def get_pose(self):
        '''
        returns:
            dict(x,y,yaw(rad))
        '''
        # data = rospy.wait_for_message('/amcl_pose',PoseWithCovarianceStamped)
        try:
            data = rospy.wait_for_message("/RosAria/pose",Odometry,2)
        except:
            return None
        x = data.pose.pose.position.x 
        y = data.pose.pose.position.y
        yaw = data.pose.pose.orientation.z
        return dict(x=x,y=y,yaw=yaw)
    
    # def get_cam_pose(self):
    #     '''
    #     returns:
    #         yaw (rad)
    #     '''
    #     data = rospy.wait_for_message('/pioneer3at/cam_position_controller/state',JointControllerState)
    #     joint_angle = data.process_value
    #     return joint_angle

    def get_cam_pose(self):
        '''
        returns:
            For now, yaw only!
            List [joint pos ID1, joint pos ID2]
            in randian ID1: yaw, ID2: roll
        '''
        data = rospy.wait_for_message('/dynamixel_workbench/joint_states',JointState)
        joint_angle = list(data.position)
        # print(joint_angle)
        return joint_angle[0]

def get_query_object(query_name):
    data = rospy.wait_for_message('/gazebo/model_states',ModelStates)
    names = data.name
    positions = data.pose
    for ind, name in enumerate(names):
        if len(name.split(query_name)) >1:
            break
    pose = positions[ind]
    query_q = pose.orientation
    query_q = euler_from_quaternion([query_q.x,query_q.y,query_q.z,query_q.w])
    yaw = query_q[2]
    x = pose.position.x
    y = pose.position.y
    pose =  dict(x=x,y=y,yaw=yaw)
    return dict(name=name,pos = pose)

def get_shortest_path_to_point(start_pose,goal_pose,tolerance=0.01):
    rospy.wait_for_service('/move_base/NavfnROS/make_plan')
    request = rospy.ServiceProxy('/move_base/NavfnROS/make_plan', GetPlan)
    start = PoseStamped()
    start.header.frame_id = 'map'
    q = quaternion_from_euler(0, 0, start_pose['yaw'])
    start.pose.orientation.x = q[0]
    start.pose.orientation.y = q[1]
    start.pose.orientation.z = q[2]
    start.pose.orientation.w = q[3]
    start.pose.position.x = start_pose['x']
    start.pose.position.y = start_pose['y']
    start.pose.position.z = 0

    goal = PoseStamped()
    goal.header.frame_id = 'map'
    q = quaternion_from_euler(0, 0, goal_pose['yaw'])
    goal.pose.orientation.x = q[0]
    goal.pose.orientation.y = q[1]
    goal.pose.orientation.z = q[2]
    goal.pose.orientation.w = q[3]
    goal.pose.position.x = goal_pose['x']
    goal.pose.position.y = goal_pose['y']
    goal.pose.position.z = 0


    resp = request(start,goal,tolerance)
    waypoints = resp.plan.poses
    res = 0
    if len(waypoints)>1:
        px = waypoints[0].pose.position.x
        py = waypoints[0].pose.position.y
        for wp in waypoints[1:]:
            x = wp.pose.position.x
            y = wp.pose.position.y
            res += math.sqrt((px-x)**2+(py-y)**2)
            px = x
            py = y
    return res