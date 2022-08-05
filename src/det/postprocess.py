import torch
import cv2
import copy

VOC_CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)
CLASS_NAME_NEW = (*VOC_CLASS_NAMES, 'unknown')

def post_process(pred):
    pred = pred['instances']._fields
    pred_boxes = pred['pred_boxes']
    scores = pred['scores']
    pred_classes = pred['pred_classes']
    pred_epis = pred['epis']
    pred_alea = pred['alea']
    index = scores>0.2
    thres = 0.2*1280*720
    index2 = pred_boxes.area()<thres
    # print(index2)
    index = index * index2

    pred_boxes = pred_boxes[index]
    pred_classes = pred_classes[index]
    pred_epis = pred_epis[index]
    pred_alea = pred_alea[index]
    return pred_boxes, pred_classes, pred_epis, pred_alea

def plot(img,pred_boxes, pred_classes,landmark=None):
    demo_image = copy.deepcopy(img)

    for bbox,label in zip(pred_boxes,pred_classes):
        if label==20:
            color = (0,255,255)
        else:
            color = (255,0,0)
        cv2.rectangle(demo_image, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]),int(bbox[3])), color, 2)
        if landmark == None:
            class_name = CLASS_NAME_NEW[int(label)]
        else:
            class_name = landmark[int(label)]
        cv2.putText(demo_image,class_name , 
                                (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return demo_image

def plot_candidate(img,pred_boxes,pred_score,query_name):
    demo_image = copy.deepcopy(img)

    for bbox,score in zip(pred_boxes,pred_score):
        color = (255,0,0)
        cv2.rectangle(demo_image, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.putText(demo_image,query_name + ' '+  str(round(score.item(),2)), 
                                (int(bbox[0]), int(bbox[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    return demo_image