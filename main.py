"""Simple proof-of-concept real-time card scanner written in Python

Implementation details were taken from the docs by PQPO at:
https://pqpo.me/2018/09/12/android-camera-real-time-scanning/"""

# External Libraries
import cv2
import numpy as np
from argparse import ArgumentParser

# Custom Libraries
from Camera import Camera

# Note: Ratio params are based off of the max_size param, actual values are in round brackets in the comments
PARAMS = {
    "max_size": 300,  # width of scaled down image for faster processing
    "frame_scaling_factor": 0.6,  # ratio of unmasked area to the entire frame (180)
    "mask_aspect_ratio": (86, 54),  # CR80 standard card size is 86mm x 54mm
    "gaussian_blur_radius": 5,  # higher radius = more blur
    "canny_lowerthreshold_ratio": 0.03,  # rejected if pixel gradient below lower threshold (9)
    "canny_upperthreshold_ratio": 0.10,  # accepted if pixel gradient above upper threshold (30)
    "dilate_kernel_size": 3,  # larger kernel = thicker lines
    "houghline_threshold_ratio": 0.5,  # minimum intersections to detect a line (150)
    "houghline_minlinelength_ratio": 0.2,  # minimum length of a line (60)
    "houghline_maxlinegap_ratio": 0.005,  # maximum gap between two points to form a line (2)
    "area_detection_ratio": 0.15,  # ratio of the detection area to the image area (45)
    "corner_quality_ratio": 0.99,  # higher value = stricter corner detection
    "wait_frames": 0,  # number of consecutive valid frames to wait
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


def create_trackbars() -> None:
    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 640, 0)
    cv2.createTrackbar(
        "Gaussian Blur Radius",
        "Parameters",
        PARAMS["gaussian_blur_radius"],
        21,
        lambda x: print(f"Gaussian Blur Radius: {x}"),
    )
    cv2.createTrackbar(
        "Canny Lower Threshold Ratio",
        "Parameters",
        int(PARAMS["canny_lowerthreshold_ratio"] * 100),
        100,
        lambda x: print(f"Canny Lower Threshold Ratio: {x / 100}"),
    )
    cv2.createTrackbar(
        "Canny Upper Threshold Ratio",
        "Parameters",
        int(PARAMS["canny_upperthreshold_ratio"] * 100),
        100,
        lambda x: print(f"Canny Upper Threshold Ratio: {x / 100}"),
    )
    cv2.createTrackbar(
        "Dilate Kernel Size",
        "Parameters",
        PARAMS["dilate_kernel_size"],
        21,
        lambda x: print(f"Dilate Kernel Size: {x}"),
    )
    cv2.createTrackbar(
        "Houghline Threshold Ratio",
        "Parameters",
        int(PARAMS["houghline_threshold_ratio"] * 100),
        100,
        lambda x: print(f"Houghline Threshold Ratio: {x / 100}"),
    )
    cv2.createTrackbar(
        "Area Detection Ratio",
        "Parameters",
        int(PARAMS["area_detection_ratio"] * 100),
        100,
        lambda x: print(f"Area Detection Ratio: {x / 100}"),
    )


def update_params() -> None:
    PARAMS["gaussian_blur_radius"] = cv2.getTrackbarPos(
        "Gaussian Blur Radius", "Parameters"
    )
    PARAMS["gaussian_blur_radius"] += int(not (PARAMS["gaussian_blur_radius"] % 2))
    PARAMS["canny_lowerthreshold_ratio"] = (
        cv2.getTrackbarPos("Canny Lower Threshold Ratio", "Parameters") / 100
    )
    PARAMS["canny_upperthreshold_ratio"] = (
        cv2.getTrackbarPos("Canny Upper Threshold Ratio", "Parameters") / 100
    )
    PARAMS["dilate_kernel_size"] = cv2.getTrackbarPos(
        "Dilate Kernel Size", "Parameters"
    )
    PARAMS["houghline_threshold_ratio"] = (
        cv2.getTrackbarPos("Houghline Threshold Ratio", "Parameters") / 100
    )
    PARAMS["area_detection_ratio"] = (
        cv2.getTrackbarPos("Area Detection Ratio", "Parameters") / 100
    )


def mask_video(img: np.ndarray, camW: int, camH: int) -> np.ndarray:
    """Apply a rectangle mask to capture only a portion of the image"""

    mask_border = (
        int(PARAMS["max_size"] * PARAMS["area_detection_ratio"]) if DEBUG else 0
    )
    maskW, maskH = PARAMS["mask_aspect_ratio"]
    maskW, maskH = (
        int(camW * PARAMS["frame_scaling_factor"]),
        int(camW * PARAMS["frame_scaling_factor"] * maskH / maskW),
    )

    mask = np.ones(img.shape, np.uint8)
    cv2.rectangle(
        mask,
        (
            int(camW / 2 - maskW / 2 + mask_border / 2),
            int(camH / 2 - maskH / 2 + mask_border / 2),
        ),
        (
            int(camW / 2 + maskW / 2 - mask_border / 2),
            int(camH / 2 + maskH / 2 - mask_border / 2),
        ),
        0,
        mask_border if DEBUG else -1,
    )
    mask_area = mask.astype(bool)

    masked_frame = img.copy()
    masked_frame[mask_area] = cv2.addWeighted(
        img, 1 - MASK_ALPHA, mask, MASK_ALPHA, gamma=0
    )[mask_area]

    img = img[
        int(camH / 2 - maskH / 2) : int(camH / 2 + maskH / 2),
        int(camW / 2 - maskW / 2) : int(camW / 2 + maskW / 2),
    ]
    img = cv2.resize(
        img,
        (
            PARAMS["max_size"],
            int(PARAMS["max_size"] / img.shape[1] * img.shape[0]),
        ),
    )

    return img, masked_frame, (maskH, maskW, mask_border)


def process_image(img: np.ndarray) -> np.ndarray:
    """Use Canny edge detector to highlight all the lines in the image"""

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if SHOW_PREVIEW:
        preview_grayscale = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(
        img, (PARAMS["gaussian_blur_radius"], PARAMS["gaussian_blur_radius"]), 0
    )

    # Apply Canny edge detection to find edges
    img = cv2.Canny(
        img,
        int(PARAMS["max_size"] * PARAMS["canny_lowerthreshold_ratio"]),
        int(PARAMS["max_size"] * PARAMS["canny_upperthreshold_ratio"]),
    )
    if SHOW_PREVIEW:
        preview_findlines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Dilate the image to strengthen lines and edges
    img = cv2.dilate(
        img,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                PARAMS["dilate_kernel_size"],
                PARAMS["dilate_kernel_size"],
            ),
        ),
    )
    if SHOW_PREVIEW:
        preview_end = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return (
        img,
        np.vstack([preview_grayscale, preview_findlines, preview_end])
        if SHOW_PREVIEW
        else None,
    )


def find_lines_and_corners(img: np.ndarray, mask_dims: tuple[int]) -> tuple[np.ndarray]:
    """Find the lines in the binarized image"""

    imgH, imgW = img.shape
    maskH, maskW, mask_border = mask_dims
    detectionH = int(imgH * PARAMS["area_detection_ratio"] / 2)
    detectionW = int(imgW * PARAMS["area_detection_ratio"] / 2)
    if DEBUG:
        detectionH = int(imgH * mask_border / maskH)
        detectionW = int(imgW * mask_border / maskW)
    shift_x, shift_y = imgW - detectionW, imgH - detectionH

    # Section off the image into 4 areas to check for lines
    region_left = img[:, :detectionW]
    region_right = img[:, shift_x:]
    region_top = img[:detectionH, :]
    region_bottom = img[shift_y:, :]

    # Use Hough Lines Transform to find lines in the image
    def get_houghlines(region: np.ndarray) -> np.ndarray:
        """Use the less efficient Hough Lines Transform to find lines in the image
        HoughLinesP() is faster but returns Cartesian instead of polar coordinates
        Because many lines will be found, return only 5 of them to reduce processing time"""
        lines = cv2.HoughLines(
            region,
            1,
            np.pi / 180,
            int(PARAMS["max_size"] * PARAMS["houghline_threshold_ratio"]),
        )
        return lines[:5] if lines is not None else np.array([])

    lines_left, lines_right, lines_top, lines_bottom = map(
        get_houghlines, [region_left, region_right, region_top, region_bottom]
    )

    # Draw the found lines on the preview image and for corner detection
    highlighted_lines = np.zeros((imgH, imgW, 3), np.uint8)
    preview_regions = np.zeros((imgH, imgW, 3), np.uint8)

    preview_regions[:, :detectionW] = cv2.cvtColor(region_left, cv2.COLOR_GRAY2BGR)
    preview_regions[:, shift_x:] = cv2.cvtColor(region_right, cv2.COLOR_GRAY2BGR)
    preview_regions[:detectionH, :] = cv2.cvtColor(region_top, cv2.COLOR_GRAY2BGR)
    preview_regions[shift_y:, :] = cv2.cvtColor(region_bottom, cv2.COLOR_GRAY2BGR)

    def draw_lines(lines: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> None:
        """Draw the lines by getting the Cartesian coordinates from the polar coordinates

        OpenCV Tutorial to Draw Lines
        https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html

        Explanation for the values 1000 added and subtracted in x1, y1, x2, y2
        https://stackoverflow.com/questions/18782873/houghlines-transform-in-opencv
        """
        if lines.any():
            for line in lines:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1, y1 = int(x0 + 1000 * (-b)) + shift_x, int(y0 + 1000 * a) + shift_y
                x2, y2 = int(x0 - 1000 * (-b)) + shift_x, int(y0 - 1000 * a) + shift_y

                cv2.line(highlighted_lines, (x1, y1), (x2, y2), RED, 1)
                cv2.line(preview_regions, (x1, y1), (x2, y2), RED, 1)

    draw_lines(lines_left)
    draw_lines(lines_right, shift_x=shift_x)
    draw_lines(lines_top)
    draw_lines(lines_bottom, shift_y=shift_y)

    def intersection(line1: np.ndarray, line2: np.ndarray) -> list[int]:
        """Finds the intersection of two lines given in Hesse normal form
        See https://stackoverflow.com/a/416559/19767101
        """
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        a, b = np.cos(theta1), np.sin(theta1)
        c, d = np.cos(theta2), np.sin(theta2)
        g = a * d - b * c
        if g == 0:  # if lines are parallel, there will be no intersection
            return None
        x0, y0 = (d * rho1 - b * rho2) / g, (-c * rho1 + a * rho2) / g
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]

    def find_corners(all_lines: tuple[np.ndarray]) -> np.ndarray:
        """Find the corners, which are the intersections between two sides' lines
        If multiple corners are found, return the average coordinates of the corners"""
        lines1, lines2 = all_lines
        if not lines1.any() or not lines2.any():
            return np.array([])

        corners = []
        for line1 in lines1:
            for line2 in lines2:
                corner = intersection(line1, line2)
                if corner is not None and 0 < corner[0] < imgW and 0 < corner[1] < imgH:
                    corners.append(corner)
        return np.array(corners).mean(axis=0) if corners else np.array([])

    corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = map(
        find_corners,
        [
            (lines_top, lines_left),
            (lines_top, lines_right),
            (lines_bottom, lines_left),
            (lines_bottom, lines_right),
        ],
    )

    # Draw the found lines and corners on a small preview window
    if SHOW_PREVIEW:
        if corner_topleft.any():
            x, y = corner_topleft.astype(int)
            cv2.circle(highlighted_lines, (x, y), 5, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 5, GREEN, -1)
        if corner_topright.any():
            x, y = corner_topright.astype(int)
            x += shift_x
            cv2.circle(highlighted_lines, (x, y), 5, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 5, GREEN, -1)
        if corner_bottomleft.any():
            x, y = corner_bottomleft.astype(int)
            y += shift_y
            cv2.circle(highlighted_lines, (x, y), 5, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 5, GREEN, -1)
        if corner_bottomright.any():
            x, y = corner_bottomright.astype(int)
            x += shift_x
            y += shift_y
            cv2.circle(highlighted_lines, (x, y), 5, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 5, GREEN, -1)
    else:
        highlighted_lines = preview_regions = None

    return (
        highlighted_lines,
        preview_regions,
        [corner_topleft, corner_topright, corner_bottomleft, corner_bottomright],
    )


def draw_user_feedback(
    img: np.ndarray,
    found_corners: list[np.ndarray],
    topleft: tuple[int],
    bottomright: tuple[int],
) -> np.ndarray:
    found_topleft, found_topright, found_bottomleft, found_bottomright = found_corners

    # Draw the lines on the main preview image
    draw_line = lambda found_corner, x1, y1, x2, y2: cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        GREEN if found_corner else RED,
        3,
    )

    draw_line(
        found_topleft or found_bottomleft,
        topleft[1],
        topleft[0],
        topleft[1],
        bottomright[0],
    )
    draw_line(
        found_topright or found_bottomright,
        bottomright[1],
        topleft[0],
        bottomright[1],
        bottomright[0],
    )
    draw_line(
        found_topright or found_topleft,
        topleft[1],
        topleft[0],
        bottomright[1],
        topleft[0],
    )
    draw_line(
        found_bottomright or found_bottomleft,
        topleft[1],
        bottomright[0],
        bottomright[1],
        bottomright[0],
    )

    # Draw the corners on the main preview image
    draw_circle = lambda found_corner, x, y: cv2.circle(
        img,
        (x, y),
        10,
        GREEN if found_corner else RED,
        -1,
    )

    draw_circle(found_topleft, topleft[1], topleft[0])
    draw_circle(found_topright, bottomright[1], topleft[0])
    draw_circle(found_bottomleft, topleft[1], bottomright[0])
    draw_circle(found_bottomright, bottomright[1], bottomright[0])

    return img


def main() -> None:
    app_name = "Card Scanner"
    print(f"Initialized {app_name}")

    # Parse command line arguments
    args = parse_args()
    video_src = args.cam if args.cam is not None else args.video
    global SHOW_PREVIEW
    SHOW_PREVIEW = bool(args.preview)

    # Initialise the video capturing object
    cam = Camera(video_src, prevent_flip=True)
    camH, camW = cam.get_frame_size()
    if SHOW_PREVIEW:
        create_trackbars()

    # Valid Frames Counter
    valid_frames = 0

    while True:
        frame_got, frame = cam.get_frame()
        if not frame_got:
            print("No frame to process")
            return

        # Update PARAMS with trackbar values
        if SHOW_PREVIEW:
            update_params()

        # Process image and get the corners
        img, masked_frame, mask_dims = mask_video(frame, camH, camW)
        img, preview_processed = process_image(img)
        preview_lines, preview_regions, corners = find_lines_and_corners(img, mask_dims)
        corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = corners

        # Check if the corners are found
        found_corners = [
            corner.any()
            for corner in [
                corner_topleft,
                corner_topright,
                corner_bottomleft,
                corner_bottomright,
            ]
        ]
        found_corners_num = sum(found_corners)

        # Corner coordinates
        topleft = int(camW / 2 - mask_dims[0] / 2), int(camH / 2 - mask_dims[1] / 2)
        bottomright = int(camW / 2 + mask_dims[0] / 2), int(camH / 2 + mask_dims[1] / 2)

        # Save card screenshot region first
        if found_corners_num >= 3:
            card = masked_frame.copy()[
                topleft[0] : bottomright[0], topleft[1] : bottomright[1]
            ]

        # Draw the user feedback
        masked_frame = draw_user_feedback(
            masked_frame, found_corners, topleft, bottomright
        )

        # Show the previews with the edges highlighted
        if SHOW_PREVIEW:
            preview = np.vstack([preview_processed, preview_lines, preview_regions])
            preview = np.pad(
                preview,
                (
                    (0, masked_frame.shape[0] - preview.shape[0]),
                    (0, 0),
                    (0, 0),
                ),
            )
            masked_frame = np.hstack([preview, masked_frame])
        cv2.imshow(app_name, masked_frame)

        # Save a screenshot if a card is detected for Y milliseconds
        if found_corners_num >= 3:
            valid_frames += 1
            if valid_frames >= PARAMS["wait_frames"]:
                valid_frames = 0
                cv2.imshow("Auto Captured Card", card)
                cv2.waitKey(0)
                cv2.destroyWindow("Auto Captured Card")
                # cv2.imwrite(f"card-{time.time()}.png", card)
        else:
            valid_frames = 0

        # Press ESC or "q" to quit
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord("q"):
            print(f"Shutting Down {app_name}...")
            cam.release()
            return


if __name__ == "__main__":
    main()
