#!/usr/bin/env python3

# importing sys
import sys

# adding Folder_2 to the system path
sys.path.insert(0, '/home/rilab/catkin_ws/src/zavis_ros/osod')

import cv2
import torch
from det.postprocess import post_process,plot, plot_candidate
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,PointCloud2
from pcl_utils import PointCloudVisualizer
import open3d as o3d

from det.matcher import matcher
from engine.predictor import DefaultPredictor
from config.config import get_cfg
from model.rcnn import GeneralizedRCNN

def load_cfg():
    cfg = get_cfg()
    cfg.merge_from_file('../osod/config_files/voc.yaml')
    cfg.MODEL.SAVE_IDX=19 #23
    cfg.MODEL.RPN.USE_MDN=False
    cfg.log = False 
    cfg.MODEL.ROI_HEADS.USE_MLN = True
    cfg.MODEL.ROI_HEADS.AUTO_LABEL = False
    cfg.MODEL.ROI_HEADS.AF = 'baseline'
    cfg.MODEL.RPN.AUTO_LABEL = False
    cfg.MODEL.ROI_BOX_HEAD.USE_FD = False
    cfg.MODEL.RPN.AUTO_LABEL_TYPE = 'sum'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.MODEL.ROI_HEADS.UNCT = True
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.PATH = '../osod'
    cfg.phase = 'voc'
    # cfg.merge_from_list(args.opts)
    RPN_NAME = 'mdn' if cfg.MODEL.RPN.USE_MDN else 'base'
    ROI_NAME = 'mln' if cfg.MODEL.ROI_HEADS.USE_MLN else 'base'
    MODEL_NAME = RPN_NAME + ROI_NAME
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg,MODEL_NAME

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        print("Loading Model")
        cfg,MODEL_NAME = load_cfg()
        device = 'cuda:0'
        model = GeneralizedRCNN(cfg,device = device).to(device)
        state_dict = torch.load('../osod/ckpt/{}/{}_{}_15000.pt'.format(cfg.MODEL.ROI_HEADS.AF,cfg.MODEL.SAVE_IDX,MODEL_NAME),map_location=device)
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(pretrained_dict)

        self.predictor = DefaultPredictor(cfg,model)
        self.matcher = matcher()
        self.query_name = "yellow book"
        self.matcher.tokenize(self.query_name)
        print("Model Loaded")
        self.pcl = PointCloudVisualizer()


    def detect(self):
        
        data = rospy.wait_for_message('/stereo_inertial_publisher/color/image',Image)
        pcl_data = rospy.wait_for_message('/stereo_inertial_publisher/stereo/points',PointCloud2)

        cv_image= self.bridge.imgmsg_to_cv2(data,"bgr8")
        pred = self.predictor(cv_image)
        
        self.pcl.convertCloudFromRosToOpen3d(pcl_data)
        # self.pcl.visualize_pcd()

        pred_boxes, pred_classes, pred_epis, pred_alea = post_process(pred)
        unk = (pred_classes ==20)
        demo_image = plot(cv_image,pred_boxes,pred_classes)
        score,candidate_box,show_patch = self.matcher.matching_score(cv_image,pred_boxes[unk])
        candidate_image= plot_candidate(cv_image,candidate_box,score,self.query_name)
        # cv2.imshow("Raw", cv_image)
        demo_image = cv2.resize(demo_image, None,  fx = 0.5, fy = 0.5) 
        candidate_image = cv2.resize(candidate_image,None, fx = 0.5, fy = 0.5) 
        cv2.imshow("Det", demo_image)
        cv2.imshow("CLIP",candidate_image)
        cv2.waitKey(3)
    
ic = image_converter()
rospy.init_node("detector")
while not rospy.is_shutdown():
    ic.detect()
    rospy.sleep(0.2)
cv2.destroyAllWindows()