# realtime video capture from realsense L515 LiDAR camera using opencv and realsense2 library

# Importing library
from cmath import pi
import pyrealsense2 as rs
import cv2
import numpy as np
import torch

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
    print(result.pandas().xyxy[0])
    
    # Show images
    cv2.imshow('frame', result.imgs[0])
    #create exit key
    if cv2.waitKey(1) == ord('q'):
        pipeline.stop()
        break


print("End of program")

