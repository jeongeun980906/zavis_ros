import rospy
from sensor_msgs.msg import JointState
from dynamixel_workbench_msgs.srv import DynamixelCommand
import argparse

def deg2command(angle):
    return int(1024*(angle/90) + 2048)

class pantilit_controller:
    def __init__(self):
        pass
    def get_cam_pose(self):
        data = rospy.wait_for_message('/dynamixel_workbench/joint_states',JointState)
        joint_angle = list(data.position)
        print(joint_angle)
        return joint_angle
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
        print(bool(res))
        return res


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles', nargs='+', type=int)
    args = parser.parse_args()
    if args.angles == None:
        cmd_angles = [0,0]
    else:
        cmd_angles = args.angles
    print('cmd angles',cmd_angles)
    controller = pantilit_controller()
    rospy.init_node('pantilit_teleop')
    while not rospy.is_shutdown():
        controller.get_cam_pose()
        controller.move_cam(cmd_angles)
        rospy.sleep(0.1)