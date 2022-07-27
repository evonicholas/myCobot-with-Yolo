# realtime video capture from realsense L515 LiDAR camera using opencv and realsense2 library

# Importing library
from cmath import pi
import pyrealsense2 as rs
import cv2
import numpy as np
import torch

# Hardware reset (Realsense frame didnt arrive problem)
ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()

# YoloV5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [47, 49] # Filtering class to apple and orange

# Importing camera parameter
camera_matrix = np.load("cameracalibration/intel_sense/matrix.npy")
distortion = np.load("cameracalibration/intel_sense/dist.npy")
t_vect = np.load("cameracalibration/intel_sense/tvecs.npy")
r_vect = np.load("cameracalibration/intel_sense/rvecs.npy")

# Realsense L515 camera settings note: some resolution might not supported check official website
L515_resolution_width = 1024 # pixels
L515_resolution_height = 768 # pixels
L515_frame_rate = 30

resolution_width = 1280 # pixels
resolution_height = 720 # pixels
frame_rate = 15  # fps

dispose_frames_for_stablisation = 30

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, L515_resolution_width, L515_resolution_height, rs.format.z16, L515_frame_rate)
config.enable_stream(rs.stream.infrared, 0, L515_resolution_width, L515_resolution_height, rs.format.y8, L515_frame_rate)
config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

pipeline.start(config)

# Flag to run code once
apple_detected = False

# Ignoring first 5 frames for better result (Recommendation)
# Skip 5 first frames to give the Auto-Exposure time to adjust
for _ in range(5):
    pipeline.wait_for_frames()

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_colormap_dim = color_image.shape

    # Undistort image
    h, w = color_image.shape[:2]
    new_cmr_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, (w,h), 1, (w,h))
    dst = cv2.undistort(color_image, camera_matrix, distortion, None, new_cmr_mtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    
    # Object detection
    result = model(dst)
    result.render()
    detected_object = result.pandas().xyxy[0]
    obj_data = result.pandas().xyxy[0]
    obj_name = obj_data.name[0]
    obj_coordinates_x = int((obj_data.xmax[0] + obj_data.xmin[0])/2) # Center X
    obj_coordinates_y = int((obj_data.ymax[0] + obj_data.ymin[0])/2) # Center Y

    # Depth of detected object

    # Only detect apple for now
    if obj_name == "apple" and apple_detected == False:
        #depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [obj_coordinates_x, obj_coordinates_y], depth)
        #print("Apple point = ", depth_point)
        print("Apple center x = ", obj_coordinates_x, "Apple center y = ", obj_coordinates_y)
        apple_detected = True

    
    # Show images
    cv2.imshow('frame', result.imgs[0])
    
    # Create exit key
    if cv2.waitKey(1) == ord('q'):
        pipeline.stop()
        break


print("End of program")
