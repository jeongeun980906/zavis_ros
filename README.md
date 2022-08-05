# ZAVIS on Pioneer 3AT
Zero-shot Active VIsual Search for Pioneer 3AT experiments repository

### Pantilit controller
```
roslaunch dynamixel_workbench_controllers dynamixel_controllers.launch
```
Test code are in 
```
python3 teleop_pantilit.py --angles 10 0
```
### Pioneer Controller
```
# Terminal2 - run rosARIA node
rosrun rosaria RosAria _port:=/dev/ttyUSB0

# If there is some issue with port,
sudo chmod a+rw /dev/ttyUSB0
```
Test code are in
```
python3 teleop_base.py
```


### LIDAR
```
rosrun sicktoolbox_wrapper sicklms _port:=/dev/ttyUSB0 _baud:=38400
```

### Camera

```
roslaunch asw_ros oakd.launch
```

Need to install [depthai-ros](https://github.com/luxonis/depthai-ros). 