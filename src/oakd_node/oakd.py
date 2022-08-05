#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime
import argparse
from vis import PointCloudVisualizer

# StereoDepth config options. TODO move to command line options
source_camera  = True
out_depth      = False  # Disparity by default
out_rectified  = True   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = False   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF # TODO

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

# TODO add API to read this from device / calib data
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
pcl_converter = PointCloudVisualizer(right_intrinsic, 1280, 720)

def create_stereo_depth_pipeline(from_camera=True):
    print("Creating Stereo Depth pipeline: ", end='')
    if from_camera:
        print("MONO CAMS -> STEREO -> XLINK OUT")
    else:
        print("XLINK IN -> STEREO -> XLINK OUT")
    pipeline = dai.Pipeline()
    
    cam          = pipeline.create(dai.node.ColorCamera)
    xout_preview = pipeline.create(dai.node.XLinkOut)
    xout_video   = pipeline.create(dai.node.XLinkOut)

    cam.setPreviewSize(1280, 720)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    xout_preview.setStreamName('rgb_preview')
    xout_video  .setStreamName('rgb_video')

    cam.preview.link(xout_preview.input)
    cam.video  .link(xout_video.input)

    if from_camera:
        cam_left      = pipeline.create(dai.node.MonoCamera)
        cam_right     = pipeline.create(dai.node.MonoCamera)
    else:
        cam_left      = pipeline.create(dai.node.XLinkIn)
        cam_right     = pipeline.create(dai.node.XLinkIn)
    stereo            = pipeline.create(dai.node.StereoDepth)
    xout_left         = pipeline.create(dai.node.XLinkOut)
    xout_right        = pipeline.create(dai.node.XLinkOut)
    xout_depth        = pipeline.create(dai.node.XLinkOut)
    xout_disparity    = pipeline.create(dai.node.XLinkOut)
    xout_rectif_left  = pipeline.create(dai.node.XLinkOut)
    xout_rectif_right = pipeline.create(dai.node.XLinkOut)

    if from_camera:
        cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        for cam in [cam_left, cam_right]: # Common config
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            #cam.setFps(20.0)
    else:
        cam_left .setStreamName('in_left')
        cam_right.setStreamName('in_right')

    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.initialConfig.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    if from_camera:
        # Default: EEPROM calib is used, and resolution taken from MonoCamera nodes
        #stereo.loadCalibrationFile(path)
        pass
    else:
        stereo.setEmptyCalibration() # Set if the input frames are already rectified
        stereo.setInputResolution(1280, 720)

    xout_left        .setStreamName('left')
    xout_right       .setStreamName('right')
    xout_depth       .setStreamName('depth')
    xout_disparity   .setStreamName('disparity')
    xout_rectif_left .setStreamName('rectified_left')
    xout_rectif_right.setStreamName('rectified_right')

    cam_left .out        .link(stereo.left)
    cam_right.out        .link(stereo.right)
    stereo.syncedLeft    .link(xout_left.input)
    stereo.syncedRight   .link(xout_right.input)
    stereo.depth         .link(xout_depth.input)
    stereo.disparity     .link(xout_disparity.input)
    if out_rectified:
        stereo.rectifiedLeft .link(xout_rectif_left.input)
        stereo.rectifiedRight.link(xout_rectif_right.input)

    streams = ['rgb_preview', 'rgb_video','left', 'right']
    if out_rectified:
        streams.extend(['rectified_left', 'rectified_right'])
    streams.extend(['disparity', 'depth'])

    return pipeline, streams

# The operations done here seem very CPU-intensive, TODO
def convert_to_cv2_frame(name, image):
    global frame_rgb
    baseline = 75 #mm
    focal = right_intrinsic[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1
    if (extended):
        max_disp *= 2
    if (subpixel):
        max_disp *= 32;
        disp_type = np.uint16  # 5 bits fractional disparity
        disp_levels = 32

    data, w, h = image.getData(), image.getWidth(), image.getHeight()
    # TODO check image frame type instead of name
    if name == 'rgb_preview':
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
        frame_rgb = frame
    elif name == 'rgb_video': # YUV NV12
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        frame_rgb = frame
    elif name == 'depth':
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == 'disparity':
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / max_disp).astype(np.uint8)

        if 1: # Optionally, apply a color map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        if pcl_converter is not None:
            if 1: # Option 1: project colorized disparity
                # print(depth.shape,frame_rgb.shape)
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                pcl_converter.rgbd_to_projection(depth, frame_rgb, True)
            else: # Option 2: project rectified right
                pcl_converter.rgbd_to_projection(depth, last_rectif_right, False)
            pcl_converter.visualize_pcd()

    else: # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name == 'rectified_right':
            last_rectif_right = frame
    return frame

def test_pipeline():
    print("Creating DepthAI device")
    with dai.Device() as device:
        cams = device.getConnectedCameras()
        depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
        
        pipeline, streams = create_stereo_depth_pipeline(source_camera)
        print("Starting pipeline")
        device.startPipeline(pipeline)

        # Create a receive queue for each stream
        q_list = []
        for s in streams:
            q = device.getOutputQueue(s, 8, blocking=False)
            q_list.append(q)
        while True:
            # Handle output streams
            for q in q_list:
                name  = q.getName()
                # print(name)
                image = q.get()
                #print("Received frame:", name)
                # Skip some streams for now, to reduce CPU load
                if name in ['rgb_preview', 'disparity']:
                    frame = convert_to_cv2_frame(name, image)
                    cv2.imshow(name, frame)
            if cv2.waitKey(1) == ord('q'):
                break
test_pipeline()