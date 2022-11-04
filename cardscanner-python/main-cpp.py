# External Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Custom Libraries
from src.Camera import Camera

# Note: Ratio params are based off of the resizedWidth param, actual values are in round brackets in the comments
PARAMS = {
    "resizedWidth": 300,  # new width of sized down image for faster processing
    "detectionAreaRatio": 0.10,  # ratio of the detection area to the image area (30)
    "sigma": 1.0,  # sigma for gaussian blur
    "cannyLowerThreshold": 10,  # rejected if pixel gradient is below lower threshold
    "cannyUpperThreshold": 30,  # accepted if pixel gradient is above upper threshold
    "houghlineThreshold": 60,  # minimum intersections to detect a line
    "houghlineMinLineLengthRatio": 0.1,  # minimum length of a line to detect (30)
    "houghlineMaxLineGapRatio": 0.1,  # maximum gap between two potential lines to join into 1 line (30)
}

RED = (0, 0, 255)
GREEN = (0, 255, 0)
MASK_ALPHA = 0.8

DEBUG = False


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Detects cards in a video or webcam feed")
    parser.add_argument(
        "--cam", type=int, default=0, help="Camera ID to use (default=0)"
    )
    parser.add_argument("--flip", type=bool, default=False, help="Flip the camera feed")
    parser.add_argument(
        "--preview", type=int, default=0, help="Show the detected edges"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Path to an image or video to process"
    )
    return parser.parse_args()


def process_image(_img: np.ndarray) -> np.ndarray:
    """Use Canny edge detector to highlight all the lines in the image"""

    # Resize the image to a smaller size for faster processing
    _img = cv2.resize(
        _img,
        (
            PARAMS["resizedWidth"],
            PARAMS["resizedWidth"] * _img.shape[0] // _img.shape[1],
        ),
    )

    # Convert to grayscale
    img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    if SHOW_PREVIEW:
        preview_grayscale = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(
        img, ksize=(0, 0), sigmaX=PARAMS["sigma"], sigmaY=PARAMS["sigma"]
    )

    # Apply Canny edge detection to find edges
    img = cv2.Canny(
        img,
        PARAMS["cannyLowerThreshold"],
        PARAMS["cannyUpperThreshold"],
    )
    if SHOW_PREVIEW:
        preview_edges = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Apply Hough line detection to find lines
    lines = cv2.HoughLinesP(
        img,
        rho=1,
        theta=np.pi / 180,
        threshold=PARAMS["houghlineThreshold"],
        minLineLength=PARAMS["resizedWidth"] * PARAMS["houghlineMinLineLengthRatio"],
        maxLineGap=PARAMS["resizedWidth"] * PARAMS["houghlineMaxLineGapRatio"],
    )

    # Draw lines on the image
    preview_findlines = np.zeros_like(preview_edges)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(preview_findlines, (x1, y1), (x2, y2), RED, 1)

    return (
        img,
        np.vstack([_img, preview_edges, preview_findlines]) if SHOW_PREVIEW else None,
    )


def main() -> None:
    # $ clear && python main-cpp.py --cam 0 --preview 1

    app_name = "Card Scanner"
    print(f"Initialized {app_name}")

    # Parse command line arguments
    args = parse_args()
    video_src = args.cam if args.cam is not None else args.video
    global SHOW_PREVIEW
    SHOW_PREVIEW = bool(args.preview)

    # Initialise the video capturing object
    cam = Camera(video_src, prevent_flip=True)

    camW, camH = cam.get_frame_size()
    print(f"Camera resolution: {camW}x{camH}")
    frameW, frameH = 480, 301
    guideW, guideH = 440, 277

    frame_got, frame = cam.get_frame()
    cropped = frame[
        camH // 2 - frameH // 2 : camH // 2 + frameH // 2,
        camW // 2 - frameW // 2 : camW // 2 + frameW // 2,
    ]
    color = ("b", "g", "r")
    for i, col in enumerate(color):
        histr = cv2.calcHist([cropped], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show(block=False)

    while True:
        frame_got, frame = cam.get_frame()
        if not frame_got:
            print("No frame to process")
            return

        # Crop image to be 480 x 301 from the center
        cropped = frame[
            camH // 2 - frameH // 2 : camH // 2 + frameH // 2,
            camW // 2 - frameW // 2 : camW // 2 + frameW // 2,
        ]

        # Process image and get the corners
        img, preview_processed = process_image(cropped)

        # Show the previews with the edges highlighted
        cv2.imshow(app_name, preview_processed)

        # Press "s" to save a screenshot, ESC or "q" to quit
        key = cv2.waitKey(1)
        if key == ord("s"):
            ...
        elif key == 27 or key == ord("q"):
            print(f"Shutting Down {app_name}...")
            cam.release()
            return


if __name__ == "__main__":
    main()
