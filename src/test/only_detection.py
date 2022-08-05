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


VOC_CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
CLASS_NAME_NEW = (*VOC_CLASS_NAMES, 'unknown')

LANDMARK_IN = [10, 17, 19] # 8
in_landmark_names = ['diningtable','sofa','tvmonitor'] #'chair',
out_landmark_names = ['desk','drawer','side table','coffee table','bed','arm chair']
landmark_names = in_landmark_names+out_landmark_names

detection_labels = []
for l in in_landmark_names:
    detection_labels.append(VOC_CLASS_NAMES.index(l))


def to_rad(th):
    return th*math.pi / 180

class det:
    def __init__(self,query_name = 'cellphone'):
        self.bridge = CvBridge()
        self.landmark_names = landmark_names
        # self.landmark_query_names = (*landmark_names,query_name)
        print("Loading Model")
        self.predictor = load_det()
        self.matcher = matcher(out_landmark_names)
        self.query_name = query_name
        self.matcher.tokenize(self.query_name)
        print("Model Loaded")
        self.PCV = PointCloudVisualizer()
        width = 848
        height = 480
        fov_x = 85.2
        fov_y = 57
        self.width = width
        self.height = height
        # camera intrinsics
        focal_length_x = 0.5 * width / math.tan(to_rad(fov_x/2))
        focal_length_y = 0.5 * height / math.tan(to_rad(fov_y/2))
        self.fx, self.fy, self.cx, self.cy = (focal_length_x,focal_length_y, width/2, height/2)
    def detect_landmarks_query(self,vis=True):
        data = rospy.wait_for_message('/stereo_inertial_publisher/color/image',Image)
        # data = rospy.wait_for_message('/camera/color/image_raw',Image)        /stereo_inertial_publisher/color/image
        # data2 = rospy.wait_for_message('/camera_controller/depth/image_raw',Image)        
        
        pcl_data = rospy.wait_for_message('/stereo_inertial_publisher/stereo/points',PointCloud2)
        # pcl_data = rospy.wait_for_message('/camera/depth/color/points',PointCloud2)
        # cpos = rospy.wait_for_message("/amcl_pose",PoseWithCovarianceStamped)

        cv_image= self.bridge.imgmsg_to_cv2(data,"bgr8")
        # cv_image_depth= self.bridge.imgmsg_to_cv2(data2,desired_encoding='passthrough')
        # print(cv_image_depth.dtype)
        # cv_image_depth = np.array(cv_image_depth, dtype=np.float32)
        pred = self.predictor(cv_image)
        # print(cv_image.shape)
        self.PCV.convertCloudFromRosToOpen3d(pcl_data)

        pred_boxes, pred_classes, pred_epis, pred_alea = post_process(pred)

        in_pred_classes = []
        mask = torch.zeros_like(pred_classes,dtype=torch.bool)
        for e,cat in enumerate(pred_classes):
            if cat in detection_labels:
                mask[e] = True
                in_pred_classes.append(detection_labels.index(cat.item()))
        in_pred_classes = torch.LongTensor(in_pred_classes)
        in_pred_boxes = pred_boxes[mask].tensor.cpu()
        in_entropy = torch.zeros(in_pred_classes.shape[0])
        mask = (pred_classes ==20)
        out_pred_boxes = pred_boxes[mask]

        demo_image = plot(cv_image,pred_boxes,pred_classes)
        out_pred_boxes, out_pred_classes, out_entropy, query_score, query_boxes,query_candidate_patchs = self.matcher.matching(cv_image,out_pred_boxes)
        if len(out_pred_classes)>0:
            out_pred_classes = out_pred_classes + len(in_landmark_names)

        pred_classes = torch.cat((in_pred_classes,out_pred_classes),axis=0)
        pred_boxes = torch.cat((in_pred_boxes,out_pred_boxes),axis=0)
        pred_entropy = torch.cat((in_entropy,out_entropy),axis=0)
        landmark_index = pred_classes.shape[0]
        # print(landmark_index)
        total_boxes = torch.cat((pred_boxes,query_boxes),axis=0)
        poses = self.transform(total_boxes)
        if vis:
            landmark_image = plot(cv_image,pred_boxes,pred_classes,self.landmark_names)
            query_image = plot_candidate(cv_image,query_boxes,query_score,self.query_name)
            demo_image = cv2.resize(demo_image, None,  fx = 0.5, fy = 0.5) 
            landmark_image = cv2.resize(landmark_image,None, fx = 0.5, fy = 0.5) 
            query_image = cv2.resize(query_image,None, fx = 0.5, fy = 0.5) 
            # cv_image_depth = cv2.resize(cv_image_depth,None, fx = 0.5, fy = 0.5) 
            # print(cv_image_depth)
            # cv2.imshow("depth", cv_image_depth)
            cv2.imshow("OSOD", demo_image)
            cv2.imshow("CLIP_Landmark",landmark_image)
            cv2.imshow("CLIP_Query",query_image)
            cv2.waitKey(3)
        return poses[:landmark_index],pred_classes,pred_entropy,poses[landmark_index:],query_candidate_patchs
    def transform(self,candidate_boxes):
        pcd = copy.deepcopy(self.PCV.pcl)
        trs = np.asarray([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]])
        colors = np.asarray(pcd.colors) # [N x 3]
        points = np.asarray(pcd.points) #[N x 3]
        points = np.dot(trs,points.T).T #[N x 3]
        points =points / points[:,-1].reshape(-1,1)
        points = points[:,:-1].astype(np.int16) # [N x 2]
        img = np.ones((self.height,self.width,3))*-100
        mapping = np.zeros((self.height,self.width,3))
        pcd.transform([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        map_points = np.asarray(pcd.points)
        # print(points.shape)
        for p,c,mp in zip(points,colors,map_points):
            img[p[1],p[0]] = c
            mapping[p[1],p[0]] = mp
        img,mapping = crop_zeros(img,mapping)
        img = cv2.resize(img,(1280,720))
        mapping = cv2.resize(mapping,(1280,720))
        # cv2.imshow("?",img)
        # o3d.visualization.draw_geometries([pcd])
        res = []
        candidate_boxes = candidate_boxes.cpu().numpy().astype(np.int16)
        for candidate_box in candidate_boxes:
            temp = []
            crop_points = copy.deepcopy(mapping[candidate_box[1]:candidate_box[3],candidate_box[0]:candidate_box[2],:])
            crop_img = copy.deepcopy(img[candidate_box[1]:candidate_box[3],candidate_box[0]:candidate_box[2],:])
            # plt.figure()
            # plt.imshow(crop_points)
            # plt.show()
            new_height = crop_points.shape[0]
            new_width = crop_points.shape[1]
            center_points = crop_points[int(2*new_height/5):int(3*new_height/5),int(2*new_width/5):int(3*new_width/5),:]
            center_points = center_points.reshape(-1,3)
            center_imgs = crop_img[int(2*new_height/5):int(3*new_height/5),int(2*new_width/5):int(3*new_width/5),:]
            # plt.figure()
            # plt.imshow(center_imgs)
            # plt.show()
            
            center_imgs = center_imgs.reshape(-1,3)
            
            mask_x = (center_imgs[:,0] > 0.0)
            mask_y = (center_imgs[:,1] > 0.0)
            mask_z = (center_imgs[:,2] > 0.0)
            mask = mask_x & mask_y & mask_z
            center_points = center_points[mask]
            # print(center_points.shape)
            for cp in center_points:
                if cp[2]< 2.0 and cp[2]>-1.0 and cp[0]<3:
                    temp.append(dict(x=cp[0],y=cp[1],z=cp[2]))
            res.append(temp)
        return res

def crop_zeros(image,mapping):
    y_nonzero, x_nonzero, _ = np.nonzero(image+100)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)], \
                mapping[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

if __name__ == '__main__':
    d = det()
    map = minimap()
    rospy.init_node('a')
    while not rospy.is_shutdown():
        lposes, llabels, lentropy,query_poses,query_patches = d.detect_landmarks_query(vis=True)
        map.color_landmarks(lposes,llabels,lentropy)
        figure = map.plot()
        cv2.imshow('map',figure)
        map.reset()
        k = cv2.waitKey(1) & 0xFF
        # map.color_query(query_poses,query_patches)
        rospy.sleep(0.1)
    cv2.destroyAllWindows()