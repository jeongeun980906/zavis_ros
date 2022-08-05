import matplotlib.pyplot as plt
import copy
import numpy as np
import yaml
import cv2

MAP_PATH = '../../maps/mymap.pgm'
config_file = '../../maps/mymap.yaml'

with open(MAP_PATH, 'rb') as pgmf:
    im = plt.imread(pgmf)

with open(config_file) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    map_config = yaml.load(file, Loader=yaml.FullLoader)
STEP_SIZE = map_config['resolution']
ORGIN = map_config['origin'][:-1]
unknown_color = int(map_config['free_thresh']*255)

map = np.zeros((im.shape[0],im.shape[1],3))
occupied = (im==0)
map[occupied] = [0,0,0]
free = (im == 254)
map[free] = [1,1,1]

POINT_SIZE = 2
MODE = 'point'
COLOR = 'black'
# callback함수
global last_x, last_y,drawing
last_x = None
last_y = None
drawing = False 
ix,iy = None,None
temp_map = copy.deepcopy(map)
new_map = copy.deepcopy(temp_map)

def callback(event, x, y, flags, param):
    global last_x,last_y,drawing,temp_map,ix,iy,new_map
    color = [0,0,0] if COLOR=='black' else [255,255,255]
    if MODE == 'point':
        if event == cv2.EVENT_LBUTTONUP:
            temp_map[y-POINT_SIZE:y+POINT_SIZE,x-POINT_SIZE:x+POINT_SIZE] = color
            new_map = copy.deepcopy(temp_map)
        if event == cv2.EVENT_MBUTTONUP:
            temp_map[y-POINT_SIZE-2:y+POINT_SIZE+2,x-POINT_SIZE-2:x+POINT_SIZE+2] = map[y-POINT_SIZE-2:y+POINT_SIZE+2,x-POINT_SIZE-2:x+POINT_SIZE+2]
            new_map = copy.deepcopy(temp_map)
    if MODE == 'rec':
        if event == cv2.EVENT_LBUTTONDOWN:
            last_x = x 
            last_y = y
            drawing = True 
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True: 
                if ix != None:
                    temp_map[last_y:iy,last_x:ix] = new_map[last_y:iy,last_x:ix]
                temp_map[last_y:y,last_x:x] = color
                ix = x
                iy = y
        if event == cv2.EVENT_LBUTTONUP and last_x!=None and last_y != None:
            temp_map[last_y:y,last_x:x] = color
            last_x = None
            last_y = None
            drawing = False
            ix,iy = None,None
            new_map = copy.deepcopy(temp_map)

cv2.namedWindow('image')
# cv2.resizeWindow(winname='image', width=2000, height=2000)
cv2.setMouseCallback('image', callback)

while(1):
    cv2.imshow('image', temp_map)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break 

    if k == ord('p'):
        MODE = 'point'

    if k == ord('e'):
        MODE = 'rec'
    
    if k == ord('w'):
        COLOR = 'white'

    if k == ord('r'):
        COLOR = 'black'
    
    if k == ord('s'):
        np.save('../../maps/new_map.npy',new_map)
        break

cv2.destroyAllWindows()