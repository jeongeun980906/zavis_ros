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

        self.landmark_names = landmark_names
        self.landmark_colors = plt.cm.get_cmap('tab20b', len(landmark_names))

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