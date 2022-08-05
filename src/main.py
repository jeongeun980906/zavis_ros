import rospy
from map_utils.map import frontier_map
from det.detector import zeroshot_det,landmark_names
from map_utils.controller import ros_controller,get_query_object,get_shortest_path_to_point
from co_occurance.comet_co import co_occurance_score
from co_occurance.schedular import traj_schedular

import numpy as np
import cv2

import copy
import matplotlib.pyplot as plt

query_name='cellphone'
scene_map = frontier_map()
detector = zeroshot_det(query_name=query_name)
controller = ros_controller()

co_occurance_scoring = co_occurance_score('cuda:0')
co_occurance_scoring.landmark_init(landmark_names)
co_occurance = co_occurance_scoring.score(query_name)
print(co_occurance,landmark_names)
rospy.init_node("asw")

# query = get_query_object('Tablet')
# cpos = controller.get_pose()
# min_dis = get_shortest_path_to_point(cpos,query['pos'])
# print("Min distance: ", min_dis)

sche = traj_schedular(landmark_names)
sche.set_score(co_occurance)
while not rospy.is_shutdown():
    '''
    detect all 180 ranges
    '''
    dis = 0
    scene_map.scan()
    detected_landmarks = []
    candidates_qposes = []
    candidate_patches = np.zeros((0,256,256,3),dtype=np.uint8)
    for i in range(3):
        controller.move_cam([(i-1)*60,0])
        rospy.sleep(0.5)
        lposes, llabels, lentropy,query_poses,query_patches = detector.detect_landmarks_query(vis=True)
        # print(len(query_poses),query_patches.shape)
        lposes,llabels,lentropy = scene_map.postprocesslandmarks(lposes,llabels,lentropy)
        detected_landmarks += llabels
        candidates_qposes += query_poses
        candidate_patches = np.concatenate((candidate_patches,query_patches),axis=0)
        scene_map.color_landmarks(lposes, llabels, lentropy)
    controller.move_cam([0,0])
    detected_landmarks = list(set(detected_landmarks))
    cpos = controller.get_pose()
    candidate_poses = []
    uncts = []
    for l in detected_landmarks:
        waypoint = scene_map.get_reachable(cpos,l)
        if waypoint[0] != None:
            data = dict(name=landmark_names[l],pos=waypoint[0],rot=waypoint[1])
            uncts.append(waypoint[3])
            candidate_poses.append(data)
    # scene_map.reset_landmark(detected_landmarks)
    frontier = scene_map.frontier_detection()
    candidate_poses += frontier
    uncts += [0]*len(frontier)
    # print(candidate_poses)
    path = sche.schedule(cpos,candidate_poses,uncts)
    print(path)
    '''
    plot map
    '''
    new_map = scene_map.plot_current_pose()
    cv2.imshow('map',new_map)
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('q'):
        break 
    rospy.sleep(0.2)
    scene_map.reset_landmark(detected_landmarks)
    # scene_map.color_query(candidates_qposes,candidate_patches)
    # for p in path[1:]:
    #     print(p,cpos)
    #     scene_map.color_path(p,cpos)
    #     if p == None:
    #         scene_map.reset()
    #         break
    #     move_sucess = controller.move2point(p['pos'],p['rot'])
    #     print('move success', move_sucess)
    #     if p['name'] == frontier:
    #         break
    #     else:
    #         for i in range(3):
    #             controller.move_cam((i-1)*30)
    #             query_poses, query_patches = detector.detect_query(vis=False)
    #             candidates_qposes += query_poses
    #             candidate_patches = np.concatenate((candidate_patches,query_patches),axis=0)
    #         controller.move_cam(0)
    # scene_map.color_query(candidates_qposes,candidate_patches)
  
cv2.destroyAllWindows()
