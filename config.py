# Configuration file. Edit as desired
import numpy as np


# Webcam number
webcam_num = 1  # system default usually is 0

# Classifier type. Options: "dnn"/"haar"/"yolo_cpu"
classifier = "dnn" # "haar"    # "yolo_cpu"   # "dnn"
# Algorithm resolution
image_ratio = 16./10.

# Size of the captured image used for face detection
capture_size = 480
capture_y = np.int(np.floor(capture_size * image_ratio))
capture_x = np.int(np.floor(capture_y * image_ratio))
detection_area_margin = 30  # subimage margin used for detection

# Canvas resolution
canvas_size = 1080
canvas_y = np.int(np.floor(canvas_size * image_ratio))
canvas_x = np.int(np.floor(canvas_y * image_ratio))

# Canvas brightness
intensity = 255  # out of 256
canvas_color = (intensity, intensity, intensity)

# Canvas window type
fullscreen = True

# Size of ellipses covering faces
ellipse_size = 1.3

# Small subframe in the upper-left corner of the videostream
subframe = True
small_subframe_offset = 20

# Threshhold for detection with DNN
thresh = 0.3
