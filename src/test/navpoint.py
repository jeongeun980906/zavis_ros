import rospy
from map_utils.controller import ros_controller

rospy.init_node("ASW")
controler = ros_controller()
cam_angle = 90
goal_pose = dict(x=2.5,y=-1.0,yaw=0)
controler.move2point(goal_pose)
while not rospy.is_shutdown():
    cam = controler.get_cam_pose()
    pos = controler.get_pose()
    print(cam,pos)
    # controler.move_cam(cam_angle)
    # controler.move2point(goal_pose)
    rospy.sleep(0.1)