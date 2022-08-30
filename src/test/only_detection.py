#!/usr/bin/env python3

# importing sys
import sys

# adding Folder_2 to the system path
sys.path.insert(0, '/home/rilab/catkin_ws/src/ASW_ros/osod')


import rospy
from det.loader import load_det
import cv2
import torch
import math
from pcl_utils import PointCloudVisualizer
from scipy import interpolate
from det.matcher import matcher
from det.postprocess import post_process,plot, plot_candidate
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt


from cv_bridge import CvBridge
from sensor_msgs.msg import Image,PointCloud2
from mini_map import minimap
from det.detector import zeroshot_det

if __name__ == '__main__':
    d = zeroshot_det()
    map = minimap()
    rospy.init_node('a')
    while not rospy.is_shutdown():
        lposes, llabels, lentropy,query_poses,query_patches = d.detect_landmarks_query(vis=True,static=True)
        lposes,llabels,lentropy = map.postprocesslandmarks(lposes,llabels,lentropy)
        map.color_landmarks(lposes,llabels,lentropy)
        figure = map.plot()
        cv2.imshow('map',figure)
        map.reset()
        k = cv2.waitKey(1) & 0xFF
        # map.color_query(query_poses,query_patches)
        rospy.sleep(0.1)
    cv2.destroyAllWindows()