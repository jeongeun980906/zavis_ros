from cmath import nan, sin
import matplotlib.pyplot as plt
import copy
import numpy as np
import cv2
import yaml
import math

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from det.detector import landmark_names
from map_utils.union import UnionFind
from tf.transformations import quaternion_from_euler, euler_from_quaternion

MAP_PATH = '../maps/mymap_new.pgm'
config_file = '../maps/mymap.yaml'

class frontier_map:
    def __init__(self):
        self.grid_map = plt.imread(MAP_PATH)
        print(self.grid_map.shape,self.grid_map.dtype)
        # self.grid_map = np.zeros((500,500),dtype=np.uint8)
        unk = (self.grid_map==255)
        self.grid_map[unk] = 130
        self.grid_map = cv2.cvtColor(self.grid_map,cv2.COLOR_GRAY2BGR)
        self.gt_map = copy.deepcopy(self.grid_map)

        with open(config_file) as file:
            map_config = yaml.load(file, Loader=yaml.FullLoader)

        self.STEP_SIZE = map_config['resolution']
        self.map_size = self.grid_map.shape
        self.scenebound = [[85,225], [180,290]]
        print(self.map_size)
        # self.ORGIN = map_config['origin'][:-1]
        self.ORGIN = [-10,-10]
        self.robot_size = 2
        self.max_laser = 5
        self.max_obj_size = 50
        self.landmark_names = landmark_names
        self.landmark_colors = plt.cm.get_cmap('Set2', len(landmark_names))

        self.unct_map = np.zeros((self.map_size[0],self.map_size[1]))

    def reset(self):
        self.grid_map = copy.deepcopy(self.gt_map)

    def scan(self):
        
        scan_data = rospy.wait_for_message("/scan",LaserScan)
        cpos = rospy.wait_for_message("/RosAria/pose",Odometry)
        cpos = cpos.pose.pose
        self.process_scan_data(scan_data,cpos)
    
    def plot_current_pose(self):
        cpos = rospy.wait_for_message("/RosAria/pose",Odometry)
        cpos = cpos.pose.pose
        gridpose = self.xyz2grid(cpos)
        temp = copy.deepcopy(self.grid_map) # H x W x 3
        temp[gridpose[0]-self.robot_size:gridpose[0]+self.robot_size,
                            gridpose[1]-self.robot_size:gridpose[1]+self.robot_size] = [0,0,255]
        show_grid = np.ones((40, temp.shape[1],3), dtype=temp.dtype)*255
        show_grid = np.concatenate((show_grid,temp),axis=0)
        for e,ln in enumerate(landmark_names):
            x = int(90* (e%4))
            y = int(20*(e//4)+10)
            # print(x,y)
            color = list(self.landmark_colors(e))[:-1] 
            color = [int(255*c) for c in color]
    
            cv2.rectangle(show_grid,(x,y),(x+5,y+5),tuple(color),-1)
            cv2.putText(show_grid,str(ln),(x+7,y+5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)
        return show_grid
        
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
                    
                    new_y = int((x-self.ORGIN[0])//self.STEP_SIZE)
                    new_x = self.map_size[0]- int((y-self.ORGIN[1])//self.STEP_SIZE)
                    if new_x < self.scenebound[0][1] and new_x>self.scenebound[0][0]:
                        if new_y>self.scenebound[1][0] and new_y<self.scenebound[1][1]:
                            color = list(self.landmark_colors(label))[:-1] 
                            color = [int(255*c) for c in color]
                            self.grid_map[new_x,new_y] = color #[255,0,0]
                            self.unct_map[new_x,new_y] = ent
                            # if sum(self.grid_map[new_x,new_y] == 0) == 3:
                            #     self.grid_map[new_x,new_y] = color
                            #     self.unct_map[new_x,new_y] = ent
        # plt.imshow(self.unct_map)
        # plt.show()
    def color_query(self,candidates_pose,candidate_patches,vis=True):
        cols = 0
        for p in candidates_pose:
            if p != []:
                cols +=1 
        plt.figure()
        i = 1
        for pose,patch in zip(candidates_pose,candidate_patches):
            if pose != []:
                imshow_grid = copy.deepcopy(self.grid_map)
                x_, y_  = np.asarray([p['x'] for p in pose]), np.asarray([p['y'] for p in pose])
                x = np.mean(x_)
                y = np.mean(y_)
                new_y = int((x-self.ORGIN[0])//self.STEP_SIZE)
                new_x = self.map_size[0]- int((y-self.ORGIN[1])//self.STEP_SIZE)
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

    def color_path(self,path,cpose):
        pose = path['pos']
        imshow_grid = copy.deepcopy(self.grid_map)
        x = pose['x']
        y = pose['y']
        new_y = int((x-self.ORGIN[0])//self.STEP_SIZE)
        new_x = self.map_size[0]- int((y-self.ORGIN[1])//self.STEP_SIZE)
        imshow_grid[new_x-3:new_x+3,new_y-3:new_y+3] = [0,255,0]

        x = cpose['x']
        y = cpose['y']
        new_y = int((x-self.ORGIN[0])//self.STEP_SIZE)
        new_x = self.map_size[0]- int((y-self.ORGIN[1])//self.STEP_SIZE)
        imshow_grid[new_x-3:new_x+3,new_y-3:new_y+3] = [255,0,0]


        plt.imshow(imshow_grid)
        plt.axis('off')
        plt.show()

    def process_scan_data(self,scan_data,cpos):
        angle_min = scan_data.angle_min
        angle_max = scan_data.angle_max
        step_size = scan_data.angle_increment
        scan_ranges = scan_data.ranges
        cgrid = self.xyz2grid(cpos)
        rot = [cpos.orientation.x, cpos.orientation.y, cpos.orientation.z, cpos.orientation.w]
        cangle = euler_from_quaternion(rot)[-1]
        # print(cangle)
        for i,data in enumerate(scan_ranges):
            angle = step_size*i+angle_min+cangle
            # print(angle)
            self.ray(data,angle,cgrid)

    
    def xyz2grid(self,pose):
        try:
            x = pose.position.x
            y = pose.position.y
        except:
            x=pose['x']
            y = pose['y']
        new_y = int((x-self.ORGIN[0])//self.STEP_SIZE)
        new_x = self.map_size[0]- int((y-self.ORGIN[1])//self.STEP_SIZE)
        return new_x,new_y
    
    def grid2xyz(self,grid_pose,yaw=0):
        x = grid_pose[1] * self.STEP_SIZE + self.ORGIN[0] 
        y = (self.map_size[0] - grid_pose[0]) * self.STEP_SIZE + self.ORGIN[1]
        return dict(x=x,y=y,yaw=yaw)

    def axis2rot(self,axis):
        theta = math.atan2(axis[1],axis[0]) # [-180, 180]
        theta = theta/math.pi*180
        theta = 270-theta
        if theta<0:
            return theta+360
        elif theta>380:
            return theta-360
        else:
            return theta

    def ray(self,range,angle,gridpose):
        if range == 'inf' or range> self.max_laser:
            range = self.max_laser
        if str(range) == 'nan':
            range = self.max_laser
        # print(range)
        STEP_LEN = range//self.STEP_SIZE
        X_STOP = math.sin(-angle)*STEP_LEN + gridpose[0]
        Y_STOP = math.cos(angle)*STEP_LEN + gridpose[1]
        x_ = np.arange(gridpose[0],X_STOP,(math.sin(-angle)+1e-2)*self.STEP_SIZE).astype(np.int16)
        y_ = np.arange(gridpose[1],Y_STOP,(math.cos(angle)+1e-2)*self.STEP_SIZE).astype(np.int16)
        for x,y in zip(x_,y_):
            if sum(self.grid_map[x,y] == 0) == 3:
                break
            self.grid_map[x,y] = [255,255,255]


    def get_reachable(self,cpos,index):
        '''
        LoI detection
        '''
        # plt.imshow(self.grid_map)
        # plt.show()
        color = list(self.landmark_colors(index))[:-1] 
        color = [int(255*c) for c in color]
        R = (self.grid_map[:,:,0]==color[0])
        G = (self.grid_map[:,:,1]==color[1])
        B = (self.grid_map[:,:,2]==color[2])
        total = R & G & B
        grid_poss = np.where(total>0)
        uncts = self.unct_map[total[:,:]]
        mean_unct = np.mean(uncts)
        try:
            x = int(np.mean(grid_poss[0]))
            y = int(np.mean(grid_poss[1]))
            self.grid_map[x-3:x+3,y-3:y+3,:] = color
        except:
            return None,None,None,None
        axiss = [[0,1],[1,0],[0,-1],[-1,0],
                [1/2,1/2],[-1/2,1/2],[-1/2,-1/2],[1/2,-1/2]]
        step_size = 20
        min_dis = 100
        res = None
        rpos = None
        raxis = None
        for axis in axiss:
            offset = self.check_reachable(x,y,axis,self.max_obj_size)
            # print(offset)
            if offset != None:
                new_x = int(x+axis[0]*offset)
                new_y = int(y+axis[1]*offset)
                # self.grid_map[new_x,new_y] = [0.5,1,0]
                new_offest = self.check_reachable2(new_x,new_y,axis,step_size)
                if new_offest != None:
                    res_x = int(x+axis[0]*(offset+new_offest))
                    res_y = int(y+axis[1]*(offset+new_offest))
                    rpos = self.grid2xyz([res_x,res_y])
                    dis = self.get_dis(rpos,cpos)
                    if dis<min_dis:
                        res = rpos
                        min_dis = dis
                        raxis = axis
        if rpos != None:
            rgrid = self.xyz2grid(rpos)
            self.grid_map[rgrid[0]-3:rgrid[0]+3,rgrid[1]-3:rgrid[1]+3] = [255,0,255]
            return res,self.axis2rot(raxis),min_dis,mean_unct
        else:
            return None,None,None,None

    def reset_landmark(self,landmark_indexs):
        for index in landmark_indexs:
            color = list(self.landmark_colors(index))[:-1] 
            color = [int(255*c) for c in color]
            R = (self.grid_map[:,:,0]==color[0])
            G = (self.grid_map[:,:,1]==color[1])
            B = (self.grid_map[:,:,2]==color[2])
            total = R & G & B
            self.grid_map[total] = copy.deepcopy(self.gt_map[total])

        color = [255,0,255]
        R = (self.grid_map[:,:,0]==color[0])
        G = (self.grid_map[:,:,1]==color[1])
        B = (self.grid_map[:,:,2]==color[2])
        total = R & G & B
        self.grid_map[total] = copy.deepcopy(self.gt_map[total])

        self.unct_map = np.zeros((self.map_size[0],self.map_size[1]))

    def check_reachable(self,x,y,axis,step_size):
        step = range(5,step_size)
        for i in step:
            x_range = list(range(x,int(x+axis[0]*i)+1)) if axis[0]>0 else list(range(int(x+axis[0]*i),x+1))
            y_range = list(range(y,int(y+axis[1]*i)+1)) if axis[1]>0 else list(range(int(y+axis[1]*i),y+1))
            new_pos = [int(x+axis[0]*(i+1)),int(y+axis[1]*(i+1))]
            if len(x_range) != len(y_range) and (len(x_range)>1 and len(y_range)>1):
                # print(len(x_range),len(y_range))
                if len(x_range)>len(y_range):
                    y_range.append(y_range[-1])
                else:
                    x_range.append(x_range[-1])
                # print(len(x_range),len(y_range))
            if x_range[-1]<self.grid_map.shape[0]-2 and x_range[0]>1 and y_range[0]>1 and y_range[-1]<self.grid_map.shape[1]-2:
                # print(sum(self.grid_map[x_range,y_range] == [130,130,130]))
                if sum(self.grid_map[x_range,y_range] == [255,255,255]).all() >0  or sum(self.grid_map[x_range,y_range] == [130,130,130]).all() >0 : #  and sum(self.grid_map[x_range,y_range] == [0.5,0.5,0.5]).all() ==0 and self.grid_map[new_pos[0],new_pos[1],0] == 1:
                    return i+1
        return None
    
    def check_reachable2(self,x,y, axis,step_size):
        for i in range(step_size,5,-1):
            x_range = list(range(x,int(x+axis[0]*i)+1)) if axis[0]>0 else list(range(int(x+axis[0]*i),x+1))
            y_range = list(range(y,int(y+axis[1]*i)+1)) if axis[1]>0 else list(range(int(y+axis[1]*i),y+1))
            if len(x_range) != len(y_range) and (len(x_range)>1 and len(y_range)>1):
                # print(len(x_range),len(y_range))
                if len(x_range)>len(y_range):
                    y_range.append(y_range[-1])
                else:
                    x_range.append(x_range[-1])
            if x_range[-1]<self.grid_map.shape[0]-1 and x_range[0]>0 and y_range[0]>0 and y_range[-1]<self.grid_map.shape[1]-1:
                # print(sum(self.grid_map[x_range,y_range] == [0,0,0]).all())
                if sum(self.grid_map[x_range,y_range] == [255,255,255]).all() >0 or sum(self.grid_map[x_range,y_range] == [130,130,130]).all() >0 :
                    return i
        return None

    def frontier_detection(self):
        img_gray_recolor = np.zeros((self.grid_map.shape[0],self.grid_map.shape[1]))
        R = (self.grid_map[:,:,0]==255)
        G = (self.grid_map[:,:,1]==255)
        B = (self.grid_map[:,:,2]==255)
        total = R & G & B
        img_gray_recolor[total] = 1
        img_gray_recolor = (img_gray_recolor*255).astype(np.uint8)

        edges = cv2.Canny(img_gray_recolor,20,10)
        index = np.where(edges != 0)
        
        frontier_map = np.zeros_like(img_gray_recolor)
        res = []
        for indx in zip(index[0],index[1]):
            if indx[0]+1<self.grid_map.shape[0]:
                right = int(self.grid_map[indx[0]+1,indx[1],0]==130)
            else:
                right = 0
            if indx[0]>0:
                left = int(self.grid_map[indx[0]-1,indx[1],0]==130)
            else:
                left= 0
            if indx[1]+1 < self.grid_map.shape[1]:
                up = int(self.grid_map[indx[0],indx[1]+1,0]==130)
            else:
                up = 0
            if indx[1]>0:
                down = int(self.grid_map[indx[0],indx[1]-1,0] == 130)
            else:
                down = 0
            center = int(self.grid_map[indx[0],indx[1],0]==130)
            if self.grid_map[indx[0],indx[1],0] != 0 and self.grid_map[indx[0],indx[1],1] != 0 and self.grid_map[indx[0],indx[1],2] != 0: 
                if left+right+up+down > 0 or center :
                    frontier_map[indx[0],indx[1]]=1
                    res.append(indx)
                
        # plt.imshow(frontier_map)
        # plt.show()
        groups = self.groupTPL(res)
        # filter_by_size = []
        # distances = []
        res = []
        for group in groups:
            if len(group)>self.robot_size:
                mean_x = sum([x[0] for x in group])/len(group)
                mean_y = sum([y[1] for y in group])/len(group)
                frontier_map[int(mean_x),int(mean_y)] = 0.5
                mean = [int(mean_x),int(mean_y)]
                mean = self.grid2xyz(mean)
                res.append(dict(name = 'frontier',pos = mean, rot = None))
        #         try:
        #             path = get_shortest_path_to_point(self.controller,cpos,mean)
        #             dis = 0
        #             last_pos = cpos
        #             for p in path:
        #                 dis += math.sqrt((last_pos['x']-p['x'])**2+(last_pos['z']-p['z'])**2)
        #                 last_pos = p
        #         except:
        #             dis = 100
                
        #         distances.append(dis)
        #         filter_by_size.append(mean)
        # del groups,img_gray_recolor,img_gray,edges
        # if len(distances)>0:
        #     sort_index = np.argsort(np.asarray(distances))
        #     return filter_by_size,sort_index
        # else:
        #     return [],None
        return res

    def groupTPL(self,TPL, distance=1):
        U = UnionFind()

        for (i, x) in enumerate(TPL):
            for j in range(i + 1, len(TPL)):
                y = TPL[j]
                if max(abs(x[0] - y[0]), abs(x[1] - y[1])) <= distance:
                    U.union(x, y)

        disjSets = {}
        for x in TPL:
            s = disjSets.get(U[x], set())
            s.add(x)
            disjSets[U[x]] = s

        return [list(x) for x in disjSets.values()]

    def get_dis(self,x1,x2):
        return math.sqrt((x1['x']-x2['x'])**2+(x1['y']-x2['y'])**2)