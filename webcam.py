import cv2
import imp
import config
import numpy as np

# testing if run on Raspberry Pi
try:
    imp.find_module('picamera')
    pi_app = True
except ImportError:
    pi_app = False

if pi_app:
    print("Using Raspberry Pi Camera Module.")
    import time
    import picamera
    from picamera.array import PiRGBArray
else:
    print("Using webcam.")


# captures camera feed and applies canvas_function to it
def feed(capture_x=1680, capture_y=720,
         canvas_x=1680, canvas_y=720,
         fullscreen=False,
         canvas_function=None):
    if pi_app:
        camera = picamera.PiCamera()
        camera.rotation = 180
        camera.resolution = (capture_x, capture_y)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(capture_x, capture_y))

        # Allowing the camera to warmup
        time.sleep(0.1)

        # Capturing the videostream
        for img in camera.capture_continuous(rawCapture,
                                             format="bgr",
                                             use_video_port=True):
            frame = np.copy(img.array)
            canvas_function(frame, canvas_x, canvas_y, fullscreen)
            rawCapture.truncate(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cap = cv2.VideoCapture(config.webcam_num)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_x)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_y)

        while(True):
            ret, frame = cap.read()
            canvas_function(frame, canvas_x, canvas_y, fullscreen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# shows simple video output
def show_frame(frame, *_):
    cv2.namedWindow('webcam videostream', cv2.WINDOW_NORMAL)
    cv2.imshow('webcam videostream', frame)


def main():
    feed(canvas_function=show_frame)


if __name__ == "__main__":
    main()
