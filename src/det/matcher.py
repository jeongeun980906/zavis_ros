import torch
import clip
import cv2
import numpy as np
from PIL import Image

class matcher:
    def __init__(self, landmark_names = None, threshold = 30, device='cuda'):
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        self.thres = threshold
        self.landmark_names = landmark_names
        self.lthres = 30
        # self.tokenize_query(query_object_name)
        self.clip_model.eval()

    def tokenize(self,query_object_name):
        text = []
        for name in self.landmark_names:
            new_name = ''
            if len(name)>2:
                for i, letter in enumerate(name):
                    if i and letter.isupper():
                        new_name += ' '
                    new_name += letter.lower()
            else:
                new_name = name
            text.append('a photo of a {}'.format(new_name))
        text.append('a photo of a {}'.format(query_object_name))
        text = clip.tokenize(text).to(self.device)
        self.text_features = self.clip_model.encode_text(text).detach().cpu()

    def matching(self,img,pred_boxes):
        boxes = pred_boxes.tensor.cpu()
        patches,vis_patches = self.make_patch(img,boxes.numpy())
        if len(patches) == 0:
            return torch.FloatTensor([]),torch.LongTensor([])
        # print(dis.shape)
        image_features = self.clip_model.encode_image(patches.to(self.device)).detach().cpu()

        dis = torch.matmul(self.text_features,image_features.T)
        dis = dis.type(torch.FloatTensor)
        landmarks = dis[:-1,:]
        softmax = torch.softmax(landmarks,0)
        # print(softmax.shape) # [6 x N]
        entropy = -torch.sum(softmax*torch.log(softmax),1)

        max_value, max_index = torch.max(landmarks,axis=0)
        index = torch.where(max_value>self.lthres)
        class_name = max_index[index]
        lboxes = boxes[index]
        # print(class_name)

        query = dis[-1,:]
        index = torch.where(query>self.thres)[0].cpu()
        qshow_patch = vis_patches[index.numpy()]
        # print(dis.shape)
        qscore = query[index]
        qboxes = boxes[index]

        return lboxes,class_name, entropy,qscore,qboxes,qshow_patch
    
    def make_patch(self,image,bboxs):
        res = []
        vis = []
        bboxs = np.asarray(bboxs,dtype=np.int16)
        for bbox in bboxs:
            y_u = bbox[2]
            y_b = bbox[0]
            x_r = bbox[3]
            x_l = bbox[1]
            crop = image[x_l:x_r,y_b:y_u,:]
            # Make boarder and resize
            y = y_u-y_b
            x = x_r-x_l
            length = max(x, y)

            top = int(length/2 - x/2)
            bottom = int(length/2 - x/2)
            left = int(length/2 - y/2)
            right = int(length/2 - y/2)

            borderType = cv2.BORDER_CONSTANT
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType)
            crop = cv2.resize(crop,(256,256))
            # convert from BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            vis.append([crop])
            # convert from openCV2 to PIL
            pil_image=Image.fromarray(crop)
            temp = self.clip_preprocess(pil_image).unsqueeze(0)
            res.append(temp)
        if len(res) == 0:
            return res,vis
        return torch.cat(res,dim=0),np.concatenate(vis,axis=0)
    
    def matching_score(self,img,pred_boxes):
        boxes = pred_boxes.tensor.cpu()

        patches,vis_patches = self.make_patch(img,boxes.numpy())
        # print(patches.shape)
        if len(patches) == 0:
            return [],[],torch.BoolTensor([False])
        image_features = self.clip_model.encode_image(patches.to(self.device)).detach().cpu()
        
        # print(text_features)
        query_features  = self.text_features[0].unsqueeze(0)
        dis = torch.matmul(query_features,image_features.T)
        # print(dis)
        index = torch.where(dis>self.thres)[1].cpu()
        # print(gt_label,index)
        show_patch = vis_patches[index.numpy()]
        # print(dis.shape)
        score = dis[0,index]
        candidate_box = boxes[index]
        return score,candidate_box,show_patch