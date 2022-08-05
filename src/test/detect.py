import rospy
from map_utils.map import frontier_map
from det.detector import zeroshot_det,landmark_names
import numpy as np
import cv2
from map_utils.controller import ros_controller

query_name='cellphone'
scene_map = frontier_map()
detector = zeroshot_det(query_name=query_name)
controller = ros_controller()

rospy.init_node("det_demo")
while not rospy.is_shutdown():
    detected_landmarks = []
    candidates_qposes = []
    candidate_patches = np.zeros((0,256,256,3),dtype=np.uint8)
    scene_map.scan()
    for i in range(3):
        controller.move_cam([(i-1)*60,0])
        rospy.sleep(0.5)
        lposes, llabels, lentropy,query_poses,query_patches = detector.detect_landmarks_query(vis=True)
        lposes,llabels,lentropy = scene_map.postprocesslandmarks(lposes,llabels,lentropy)
    
        detected_landmarks += llabels
        candidates_qposes += query_poses
        candidate_patches = np.concatenate((candidate_patches,query_patches),axis=0)
        scene_map.color_landmarks(lposes, llabels, lentropy)
    controller.move_cam([0,0])
    new_map = scene_map.plot_current_pose()
    cv2.imshow('map',new_map)
    scene_map.reset_landmark(detected_landmarks)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break 
    rospy.sleep(0.2)
cv2.destroyAllWindows()