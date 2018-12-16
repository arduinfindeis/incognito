import webcam
import facedetection
import config


def main():
    print("incognito started.")
    webcam.feed(config.capture_x,
                config.capture_y,
                config.canvas_x,
                config.canvas_y,
                config.fullscreen,
                facedetection.draw_canvas)


if __name__ == "__main__":
    main()
