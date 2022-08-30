#!/usr/bin/env python3

# importing sys
import sys

# adding Folder_2 to the system path
sys.path.insert(0, '/home/rilab/catkin_ws/src/zavis_ros/osod')

import torch
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

def load_det(device='cuda:0'):
    cfg,MODEL_NAME = load_cfg()
    model = GeneralizedRCNN(cfg,device = device).to(device)
    state_dict = torch.load('../osod/ckpt/{}/{}_{}_15000.pt'.format(cfg.MODEL.ROI_HEADS.AF,cfg.MODEL.SAVE_IDX,MODEL_NAME),map_location=device)
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(pretrained_dict)

    predictor = DefaultPredictor(cfg,model)
    return predictor