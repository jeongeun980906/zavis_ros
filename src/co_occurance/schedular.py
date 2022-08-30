from turtle import distance
import numpy as np
import math
from map_utils.controller import get_shortest_path_to_point

class traj_schedular():
    def __init__(self,landmark_names,co_thres = 0.2):
        '''
        buffer: {name, pos, rot} only landmarks
        '''
        self.buffer = [] 

        self.thres = 2 # 1
        self.ratio = 20
        self.unct_ratio = 10.0

        self.co_thres = co_thres
        self.unct_thres = 2.0
        self.landmark_names = landmark_names
        
    def set_score(self,co_occurence):
        self.co_occurence_score = co_occurence

    def schedule(self,cpos, candidate_trajs,uncts):
        register = [dict(name='current',pos=cpos,rot = cpos['yaw'])]
        frontiers = []
        score = [-1]
        unct =[0]
        for c,u in zip(candidate_trajs,uncts):
            if c['name'] != 'frontier':
                flag = True
                for w in self.buffer:
                    ncost = int(w['name'] != c['name'] )
                    if w['rot']==None:
                        rcost = 0
                    else:
                        rcost = (abs(w['rot']- c['rot']))%181
                    dcost = self.get_dis(w['pos'],c['pos'])
                    cost = dcost+ 0.01*rcost+ncost
                    # print(cost,dcost,rcost,ncost)
                    if cost<self.thres:
                        flag = False
                if flag:
                    co_score = self.landmark_names.index(c['name'])
                    co_score = self.co_occurence_score[co_score]
                    if co_score > self.co_thres and u<self.unct_thres:
                        register.append(c)
                        unct.append(u)
                        score.append(co_score) 
            else:
                frontiers.append(c)
        dis_matrix = self.distance(register)
        sorted_indx = self.optimize(score,unct,dis_matrix)
        path = []
        for idx in sorted_indx:
            path.append(register[idx])
        
        min_dis = 100
        last_pos = path[-1]['pos']
        visit_frontier = None
        for f in frontiers:
            # print(f['pos'],last_pos)
            dis = self.shortest_path_length(f['pos'],last_pos)
            if dis<min_dis:
                min_dis = dis
                visit_frontier = f
        path.append(visit_frontier)
        self.buffer += register
        if visit_frontier != None:
            self.buffer += [visit_frontier]
        return path

    def distance(self,buffer):
        score_matrix = np.eye(len(buffer))*100
        for i in range(len(buffer)):
            for j in range(len(buffer)):
                if j>=i: 
                    break
                pos1 = buffer[i]['pos']
                pos2 = buffer[j]['pos']
                dis = self.shortest_path_length(pos1,pos2)
                score_matrix[i,j] = dis
                score_matrix[j,i] = dis
        print(score_matrix)
        return score_matrix


    def optimize(self,score, unct,dis_matrix):
        index = [0]
        distance = dis_matrix
        score = self.ratio*(1+1e-3-np.asarray(score))  + self.unct_ratio * np.asarray(unct)
        for i in range(1,len(score)):
            temp = np.arange(len(score))
            temp = np.delete(temp,index)
            dis = distance.copy()[index[-1]] # [N-i x 1]
            dis = np.delete(dis,index) # [N-i-1 x 1]
            scaled_score = np.delete(score,index)
            scaled_score = dis+scaled_score # [N -i -1 x 1]
            new = np.argmin(scaled_score)
            new = temp[new]
            index.append(new)
        return index

    def shortest_path_length(self,goal,init):
        try:
            distance = get_shortest_path_to_point(init, goal,
                        tolerance=0.01
                    )
        except:
            return 100
        return distance

    def get_dis(self,x1,x2):
        return math.sqrt((x1['x']-x2['x'])**2+(x1['y']-x2['y'])**2)