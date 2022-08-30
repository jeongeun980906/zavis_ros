import rospy
from map_utils.map import frontier_map
from det.detector import zeroshot_det,landmark_names
from map_utils.controller import ros_controller,get_query_object,get_shortest_path_to_point
from co_occurance.comet_co import co_occurance_score
from co_occurance.schedular import traj_schedular

import numpy as np
import cv2

import copy
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--target',default = 'book', type=str,help='query name')
parser.add_argument('--dis_only',action='store_true' ,default=False,help='no co-occurance')
parser.add_argument('--no_unct',action='store_true' ,default=False,help='unct ratio')
args = parser.parse_args()

query_name= args.target#'book'#'cup'#'cellphone' #'tumbler'
scene_map = frontier_map()
detector = zeroshot_det(query_name=query_name)
controller = ros_controller()

if args.dis_only:
    co_occurance = [0]*len(landmark_names)
    co_thres = -1
else:
    co_occurance_scoring = co_occurance_score('cuda:0')
    co_occurance_scoring.landmark_init(landmark_names)
    co_occurance = co_occurance_scoring.score(query_name)
    co_thres = 0.2
print(co_occurance,landmark_names)

sche = traj_schedular(landmark_names,co_thres = co_thres)
sche.set_score(co_occurance)

NUM_POINTS = 0
rospy.init_node("zavis")
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
        rospy.sleep(1)
        lposes, llabels, lentropy,query_poses,query_patches = detector.detect_landmarks_query(vis=True)
        # print(len(query_poses),query_patches.shape)
        lposes,llabels,lentropy = scene_map.postprocesslandmarks(lposes,llabels,lentropy)
        # query_poses,query_patches = scene_map.postprocessquery(query_poses,query_patches)
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
        co_score = co_occurance[l]
        num_waypoint = 1 #2 if co_score>0.8 else 1
        waypoint_poses, waypoint_rots, mean_unct  = scene_map.get_reachable(cpos,l,NUM=num_waypoint)
        if waypoint_poses != None:
            for wpose, wrot in zip(waypoint_poses,waypoint_rots):
                data = dict(name=landmark_names[l],pos=wpose,rot=wrot)
                if args.no_unct:
                    uncts.append(0)
                else:
                    uncts.append(mean_unct)
                candidate_poses.append(data)
    print(uncts,detected_landmarks)
    # scene_map.reset_landmark(detected_landmarks)
    scene_map.plot_unct()
    frontier = scene_map.frontier_detection()
    candidate_poses += frontier
    uncts += [0]*len(frontier)
    # print(candidate_poses)
    path = sche.schedule(cpos,candidate_poses,uncts)
    print(path)
    '''
    plot map
    '''
    # new_map = scene_map.plot_current_pose()
    
    # cv2.imshow('map',new_map)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break 
    rospy.sleep(0.2)
    imshow_grid = scene_map.color_path(cpos,NUM_POINTS=NUM_POINTS)
    
    scene_map.reset_landmark(detected_landmarks)
    scene_map.color_query(candidates_qposes,candidate_patches)
    for p in path[1:]:
        print(p)
        candidates_qposes = []
        candidate_patches = np.zeros((0,256,256,3),dtype=np.uint8)
        if p == None:
            scene_map.reset()
            break
        NUM_POINTS += 1
        move_sucess = controller.move2point(p['pos'],p['rot'],imshow_grid,scene_map,NUM_POINTS)
        cpos = controller.get_pose()
        # scene_map.color_path(cpos,imshow_grid,local_traj,NUM_POINTS)
        print('move success', move_sucess)
        if p['name'] == frontier:
            break
        else:
            for i in range(3):
                for j in range(2):
                    controller.move_cam([(i-1)*30,-(j%2)*15])
                    rospy.sleep(1)
                    query_poses, query_patches = detector.detect_query(vis=True)
                    candidates_qposes += query_poses
                    candidate_patches = np.concatenate((candidate_patches,query_patches),axis=0)
            controller.move_cam([0,0])
        scene_map.color_query(candidates_qposes,candidate_patches)
  
cv2.destroyAllWindows()
