#!/usr/bin/env python3

# importing sys
import sys

# adding Folder_2 to the system path
sys.path.insert(0, '/home/rilab/catkin_ws/src/asw_ros/osod')

from det.loader import load_det
import cv2
import torch
import math

from scipy import interpolate
from det.matcher import matcher
from det.postprocess import post_process,plot, plot_candidate

import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image,PointCloud2,JointState
from geometry_msgs.msg import PoseWithCovarianceStamped
from control_msgs.msg import JointControllerState
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from pcl_utils import PointCloudVisualizer
import open3d as o3d
import numpy as np
import copy

VOC_CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
CLASS_NAME_NEW = (*VOC_CLASS_NAMES, 'unknown')

LANDMARK_IN = [10, 17, 19] # 8
in_landmark_names = ['diningtable','sofa','tvmonitor'] #'chair',
out_landmark_names = ['desk','coffee table','armchair']
landmark_names = in_landmark_names+out_landmark_names

detection_labels = []
for l in in_landmark_names:
    detection_labels.append(VOC_CLASS_NAMES.index(l))

def to_rad(th):
    return th*math.pi / 180

class zeroshot_det:
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
        width = 1280 #848
        height = 720 # 480
        fov = 81 # 85.2
        # fov_y = 68.8 # 57
        self.width = width
        self.height = height
        # camera intrinsics
        focal_length = 0.5 * width / math.tan(to_rad(fov/2))
        # focal_length_y = 0.5 * height / math.tan(to_rad(fov_y/2))
        self.fx, self.fy, self.cx, self.cy = (focal_length,focal_length, width/2, height/2)

    def detect_landmarks_query(self,vis=True):
        candidate_poses = []
        candidate_labels = []
        candidate_entropys = []
        candidate_qposes = []
        candidate_qpatchs = np.zeros((0,256,256,3),dtype=np.uint8)

        for _ in range(3):
            data = rospy.wait_for_message('/stereo_inertial_publisher/color/image',Image)
            # data = rospy.wait_for_message('/camera/color/image_raw',Image)
            # data2 = rospy.wait_for_message('/camera_controller/depth/image_raw',Image)        
            
            pcl_data = rospy.wait_for_message('/stereo_inertial_publisher/stereo/points',PointCloud2)
            # pcl_data = rospy.wait_for_message('/camera/depth/color/points',PointCloud2)

            # cpos = rospy.wait_for_message("/amcl_pose",PoseWithCovarianceStamped)
            cpos = rospy.wait_for_message("/RosAria/pose",Odometry)
            yaw = rospy.wait_for_message('/dynamixel_workbench/joint_states',JointState)
            yaw = yaw.position[0]
            cpos = cpos.pose.pose

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
            poses = self.transform(total_boxes,cpos,yaw)
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
            candidate_poses += poses[:landmark_index]
            candidate_labels += pred_classes.numpy().tolist()
            candidate_entropys += pred_entropy.numpy().tolist()
            candidate_qposes += poses[landmark_index:]
            candidate_qpatchs = np.concatenate((candidate_qpatchs,query_candidate_patchs),axis=0)
        return candidate_poses,candidate_labels,candidate_entropys,candidate_qposes,candidate_qpatchs

    def detect_query(self,vis=False):
        candidate_qposes = []
        candidate_qpatches = np.zeros((0,256,256,3),dtype=np.uint8)
        for _ in range(3):
            # data = rospy.wait_for_message('/camera/color/image_raw',Image)  
            data = rospy.wait_for_message('/stereo_inertial_publisher/color/image',Image)    
            # data2 = rospy.wait_for_message('/camera_controller/depth/image_raw',Image)        
            pcl_data = rospy.wait_for_message('/stereo_inertial_publisher/stereo/points',PointCloud2)
            # pcl_data = rospy.wait_for_message('/camera/depth/color/points',PointCloud2)

            # cpos = rospy.wait_for_message("/amcl_pose",PoseWithCovarianceStamped)
            cpos = rospy.wait_for_message("/RosAria/pose",Odometry)
            yaw = rospy.wait_for_message('/dynamixel_workbench/joint_states',JointState)
            yaw = yaw.process_value
            cpos = cpos.pose.pose

            cv_image= self.bridge.imgmsg_to_cv2(data,"bgr8")
            # cv_image_depth= self.bridge.imgmsg_to_cv2(data2,desired_encoding='passthrough')
            # print(cv_image.shape)
            # cv_image_depth = np.array(cv_image_depth, dtype=np.float32)
            pred = self.predictor(cv_image)

            self.PCV.convertCloudFromRosToOpen3d(pcl_data)

            pred_boxes, pred_classes, pred_epis, pred_alea = post_process(pred)
            mask = (pred_classes ==20)
            score,candidate_box,show_patch = self.matcher.matching_score(cv_image,pred_boxes[mask])
            poses = self.transform(candidate_box,cpos,yaw)
            if vis:
                query_image = plot_candidate(cv_image,candidate_box,score,self.query_name)
                query_image = cv2.resize(query_image,None, fx = 0.5, fy = 0.5) 
                cv2.imshow("CLIP_Query",query_image)
                cv2.waitKey(3)
            candidate_qposes += poses
            candidate_qpatchs = np.concatenate((candidate_qpatchs,show_patch),axis=0)
        return candidate_qposes, candidate_qpatchs

    def transform(self,candidate_boxes,pose,cam_yaw):
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
        if pose != None:
            pos = pose.position
            # rot = pose.orientation.z
            rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rot = euler_from_quaternion(rot)[-1]+ cam_yaw
            # print(rot,pos)
            pcd.transform([[math.cos(rot), -math.sin(rot),0, pos.x+0.1],
                [math.sin(rot), math.cos(rot), 0, pos.y],
                [0, 0, 1, pos.z+1.0],
                [0, 0, 0, 1]])
        map_points = np.asarray(pcd.points)
        # print(points.shape)
        for p,c,mp in zip(points,colors,map_points):
            img[p[1],p[0]] = c
            mapping[p[1],p[0]] = mp
        img,mapping = crop_zeros(img,mapping)
        img = cv2.resize(img,(1280,720))
        mapping = cv2.resize(mapping,(1280,720))
        cv2.imshow("?",img)
        # o3d.visualization.draw_geometries([pcd])
        res = []
        candidate_boxes = candidate_boxes.cpu().numpy().astype(np.int16)
        for candidate_box in candidate_boxes:
            temp = []
            crop_points = copy.deepcopy(mapping[candidate_box[1]:candidate_box[3],candidate_box[0]:candidate_box[2],:])
            crop_img = copy.deepcopy(img[candidate_box[1]:candidate_box[3],candidate_box[0]:candidate_box[2],:])
            new_height = crop_points.shape[0]
            new_width = crop_points.shape[1]
            center_points = crop_points[int(1*new_height/5):int(4*new_height/5),int(1*new_width/5):int(4*new_width/5),:]
            # center_points = crop_points
            center_points = center_points.reshape(-1,3)
            center_imgs = crop_img[int(1*new_height/5):int(4*new_height/5),int(1*new_width/5):int(4*new_width/5),:]
            # plt.figure()
            # plt.imshow(center_imgs)
            # plt.show()
            # center_imgs = crop_img
            center_imgs = center_imgs.reshape(-1,3)
            
            mask_x = (center_imgs[:,0] > 0)
            mask_y = (center_imgs[:,1] > 0)
            mask_z = (center_imgs[:,2] > 0)
            mask = mask_x & mask_y & mask_z
            center_points = center_points[mask]
            # print(center_points.shape)
            for cp in center_points:
                depth = distance(pos,cp)
                # print(depth,pos,cp)
                if depth<7:
                    if cp[2]< 2.0 and cp[2]>-0.1: # and cp[0]<5:
                        temp.append(dict(x=cp[0],y=cp[1],z=cp[2]))
            res.append(temp)
        return res

def crop_zeros(image,mapping):
    y_nonzero, x_nonzero, _ = np.nonzero(image+100)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)], \
                mapping[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

def distance(pos1,pos2):
    '''
    args:
        pos1: gemoetry msg
        pose: list
    '''
    cx = pos1.x
    cy = pos1.y
    cz = pos1.z
    return math.sqrt((cx-pos2[0])**2+(cy-pos2[1])**2+(cz-pos2[2])**2)