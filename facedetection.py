from __future__ import division
import cv2
import numpy as np
import config

# Setting some variables with config.py input
x_offset = y_offset = config.small_subframe_offset
detection_area_margin = config.detection_area_margin
detection_area_margin_x = int(np.floor(4./3. * detection_area_margin))


# haar cascade classifier
def haar_cascade_classifier(frame):
    frame_small = frame[
        detection_area_margin:frame.shape[0]-detection_area_margin,
        detection_area_margin_x:frame.shape[1]-detection_area_margin_x]

    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    detection = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')

    return detection.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5), frame_small


# deep neural network classifier
def dnn_classifier(frame):
    frame_small = frame[
        detection_area_margin:frame.shape[0]-detection_area_margin,
        detection_area_margin_x:frame.shape[1]-detection_area_margin_x]
    (h, w) = frame_small.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame_small, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt.txt',
                                   'models/res10_300x300_ssd_iter_140000.'
                                   'caffemodel')

    net.setInput(blob)
    detections = net.forward()

    num_faces = range(0, detections.shape[2])
    faces = []

    for i in num_faces:
        confidence = detections[0, 0, i, 2]

        if confidence < 0.7:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        faces.append([startX, startY, endX-startX, endY-startY])

    return faces, frame_small


def initialize_canvas(frame):
    canvas = np.zeros((frame.shape[0] - 2 * detection_area_margin,
                      frame.shape[1] - 2 * detection_area_margin_x, 3),
                      np.uint8)
    cv2.rectangle(canvas, (0, 0),
                  (frame.shape[1], frame.shape[0]),
                  config.canvas_color, -1)
    return canvas


def create_window(canvas, faces, fullscreen):
    if (fullscreen):
        cv2.namedWindow("incognito canvas", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("incognito canvas", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow("incognito canvas", cv2.WINDOW_NORMAL)

    cv2.putText(canvas, "incognito v0.1 - Nr. of faces detected: "
                + str(len(faces)),
                (x_offset, canvas.shape[0]-y_offset),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.imshow('incognito canvas', canvas)


# the main drawing function
def draw_canvas(frame, canvas_x, canvas_y, fullscreen):

    if(config.classifier == "haar"):
        faces, frame_small = haar_cascade_classifier(frame)
    elif(config.classifier == "dnn"):
        faces, frame_small = dnn_classifier(frame)

    canvas = initialize_canvas(frame)

    for (x, y, w, h) in faces:
        # drawing a rectangle on the image (frame)
        cv2.rectangle(frame,
                      (x + detection_area_margin_x,
                       y + detection_area_margin),
                      (x + w + detection_area_margin_x,
                       y + h + detection_area_margin),
                      (255, 255, 255), 2)
        # determining the center of the elipse
        center_x = int(np.floor(x + float(w)/2))
        center_y = int(np.floor(y + float(h)/2))
        # drawing the ellipse onto the image
        cv2.ellipse(canvas, (center_x, center_y),
                    (int(w * 0.7*config.ellipse_size),
                    int(h*config.ellipse_size)),
                    0, 0, 360, (0, 0, 0), -1)

    cv2.rectangle(frame, (detection_area_margin_x, detection_area_margin),
                  (frame.shape[1]-detection_area_margin_x,
                   frame.shape[0]-detection_area_margin),
                  (100, 100, 255), 2)

    canvas = cv2.resize(canvas, (canvas_x, canvas_y))

    # including the actual image ("frame") in the canvas
    if (config.subframe):
        canvas[y_offset:y_offset+frame.shape[0],
               x_offset:x_offset+frame.shape[1]] = frame

    create_window(canvas, faces, fullscreen)
