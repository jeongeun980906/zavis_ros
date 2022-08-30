import numpy as np
import cv2
import matplotlib.pyplot as plt

import copy
from det.detector import landmark_names

class minimap:
    def __init__(self):
        self.STEP_SIZE = 0.1
        self.gridmap = np.zeros((200,200,3),dtype=np.uint8)
        self.unct_map = np.zeros((200,200))
        self.ORIGIN  = [-10,-1]
        self.map_size = self.gridmap.shape
        self.landmark_names = landmark_names
        self.landmark_colors = plt.cm.get_cmap('tab20b', len(landmark_names))

    def postprocesslandmarks(self,bposes,blabels,bentropy):
        temp_map = np.zeros((len(self.landmark_names),self.map_size[0],self.map_size[1]))
        entropy_map = np.zeros((len(self.landmark_names),self.map_size[0],self.map_size[1]))
        for poses, label,entropy in zip(bposes,blabels,bentropy):
            for p in poses:
                if p != None:
                    x = p['x']
                    y = p['y']

                    new_y = int((x-self.ORGIN[0])//self.STEP_SIZE)
                    new_x = self.map_size[0]- int((y-self.ORGIN[1])//self.STEP_SIZE)
                    temp_map[label,new_x,new_y] += 1
                    entropy_map[label,new_x,new_y] += entropy
        indexes = list(np.where(temp_map>1))
        resl = np.unique(indexes[0])
        res_pose = [[] for _ in range(len(resl))]
        res_entropy = [0]*len(resl)
        res_count = [0]*len(resl)

        for label,x,y in zip(indexes[0],indexes[1],indexes[2]):
            num_count = temp_map[label,x,y]
            avg_entropy = entropy_map[label,x,y]/num_count
            if label in resl:
                ind = np.where(resl == label)[0][0]
                res_pose[ind].append(self.grid2xyz([x,y]))
                res_entropy[ind] += avg_entropy
                res_count[ind] +=1
        res_entropy = [e/n for e,n in zip(res_entropy,res_count)]
        return res_pose,resl.tolist(),res_entropy

    def color_landmarks(self,poses,labels,entropy):
        for pose,label,ent in zip(poses,labels,entropy):
            for p in pose:
                if p != None:
                    x = p['x']
                    y = p['y']
                    new_y = self.gridmap.shape[0]- int((y-self.ORIGIN[0])//self.STEP_SIZE)
                    new_x = self.gridmap.shape[1]- int((x-self.ORIGIN[1])//self.STEP_SIZE)
                    
                    color = list(self.landmark_colors(label.item()))[:-1] 
                    color = [int(255*c) for c in color]
                    self.gridmap[new_x,new_y] = color #[255,0,0]
                    self.unct_map[new_x,new_y] = ent.item()
        
    
    def color_query(self,candidates_pose,candidate_patches):
        cols = 0
        for p in candidates_pose:
            if p != []:
                cols +=1 
        print(cols)
        plt.figure()
        i = 1
        for pose,patch in zip(candidates_pose,candidate_patches):
            if pose != []:
                imshow_grid = copy.deepcopy(self.gridmap)
                x_, y_  = np.asarray([p['x'] for p in pose]), np.asarray([p['y'] for p in pose])
                x = np.mean(x_)
                y = np.mean(y_)
                new_y = int((y-self.ORIGIN[0])//self.STEP_SIZE)
                new_x = self.gridmap.shape[0]- int((x-self.ORIGIN[1])//self.STEP_SIZE)
                imshow_grid[new_x-5:new_x+5,new_y-5:new_y+5] = [0,0,255]
                plt.subplot(2,cols,i)
                plt.imshow(patch)
                plt.axis('off')
                plt.subplot(2,cols,i+cols)
                plt.imshow(imshow_grid)
                plt.axis('off')
                i += 1
                del imshow_grid
        plt.show()

    def reset(self):
        self.gridmap = np.zeros((200,200,3),dtype=np.uint8)
        self.unct_map = np.zeros((200,200))

    def plot(self):
        figure = copy.deepcopy(self.gridmap)
        cpos_y = self.gridmap.shape[0] - int((0-self.ORIGIN[0])//self.STEP_SIZE)
        cpos_x = self.gridmap.shape[0]- int((0-self.ORIGIN[1])//self.STEP_SIZE)
        figure[cpos_x-3:cpos_x+3,cpos_y-3:cpos_y+3] = [0,0,255]
        # figure = np.rot90(figure)
        return figure