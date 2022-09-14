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
    "y_milliseconds": 200,  # number of milliseconds to wait for valid frames
}

FPS = 15
WAIT_FRAMES = int(PARAMS["y_milliseconds"] / 1000 * FPS)

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
                PARAMS["dilate_structing_element_size"],
                PARAMS["dilate_structing_element_size"],
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

    # Use Probabilistic Hough Transform to find lines in the image
    get_houghlines = lambda region: cv2.HoughLinesP(
        region,
        1,
        np.pi / 180,
        int(PARAMS["max_size"] * PARAMS["houghlines_threshold_ratio"]),
        np.array([]),
        int(PARAMS["max_size"] * PARAMS["houghlines_min_line_length"]),
        int(PARAMS["max_size"] * PARAMS["houghlines_max_line_gap"]),
    )

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
        if lines is not None and lines.any():
            for line in lines:
                for x1, y1, x2, y2 in line:
                    x1, x2 = x1 + shift_x, x2 + shift_x
                    y1, y2 = y1 + shift_y, y2 + shift_y
                    cv2.line(highlighted_lines, (x1, y1), (x2, y2), RED, 1)
                    cv2.line(preview_regions, (x1, y1), (x2, y2), RED, 2)

    draw_lines(lines_left)
    draw_lines(lines_right, shift_x=shift_x)
    draw_lines(lines_top)
    draw_lines(lines_bottom, shift_y=shift_y)

    # Section off the image into the 4 corner areas to check for card corners
    corner_frame = cv2.cvtColor(highlighted_lines, cv2.COLOR_BGR2GRAY)
    region_topleft = corner_frame[:detectionH, :detectionW]
    region_topright = corner_frame[:detectionH, shift_x:]
    region_bottomleft = corner_frame[shift_y:, :detectionW]
    region_bottomright = corner_frame[shift_y:, shift_x:]

    # Use Shi-Tomasi corner detection to find corners in the image
    get_corners = lambda region: cv2.goodFeaturesToTrack(
        region, 1, PARAMS["corner_quality_ratio"], 20
    )

    corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = map(
        get_corners,
        [region_topleft, region_topright, region_bottomleft, region_bottomright],
    )

    # Draw the found lines and corners on a small preview window
    if SHOW_PREVIEW:
        if corner_topleft is not None:
            x, y = corner_topleft.astype(int).ravel()
            cv2.circle(highlighted_lines, (x, y), 5, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 5, GREEN, -1)
        if corner_topright is not None:
            x, y = corner_topright.astype(int).ravel()
            x += shift_x
            cv2.circle(highlighted_lines, (x, y), 5, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 5, GREEN, -1)
        if corner_bottomleft is not None:
            x, y = corner_bottomleft.astype(int).ravel()
            y += shift_y
            cv2.circle(highlighted_lines, (x, y), 5, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 5, GREEN, -1)
        if corner_bottomright is not None:
            x, y = corner_bottomright.astype(int).ravel()
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


def check_lines(lines: np.ndarray, min_length: int, is_vertical: bool) -> bool:
    """Check if the lines are long enough and angled correctly to count as a line"""

    if lines is None or not lines.any():
        return False

    for line in lines:
        for x1, y1, x2, y2 in line:
            width, height = abs(x2 - x1), abs(y2 - y1)

            dist = width**2 + height**2
            if dist < min_length**2:
                continue

            if x1 == x2:
                return True

            angle = np.arctan2(height, width) * 180 / np.pi
            if is_vertical:
                if abs(90 - angle) < PARAMS["angle_threshold"]:
                    return True
            else:
                if abs(angle) < PARAMS["angle_threshold"]:
                    return True
    return False


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

    # Valid Frames Counter
    valid_frames = 0

    while True:
        frame_got, frame = cam.get_frame()
        if not frame_got:
            print("No frame to process")
            return

        # Process image and get the corners
        img, masked_frame, mask_dims = mask_video(frame, camH, camW)
        img, preview_processed = process_image(img)
        preview_lines, preview_regions, corners = find_lines_and_corners(img, mask_dims)
        corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = corners

        # Check if the corners are found
        found_topleft = corner_topleft is not None and corner_topleft.any()
        found_topright = corner_topright is not None and corner_topright.any()
        found_bottomleft = corner_bottomleft is not None and corner_bottomleft.any()
        found_bottomright = corner_bottomright is not None and corner_bottomright.any()
        found_corners = [
            found_topleft,
            found_topright,
            found_bottomleft,
            found_bottomright,
        ]
        found_corners_num = sum(found_corners)

        # Corner coordinates
        topleft = int(camW / 2 - mask_dims[0] / 2), int(camH / 2 - mask_dims[1] / 2)
        bottomright = int(camW / 2 + mask_dims[0] / 2), int(camH / 2 + mask_dims[1] / 2)

        # Save a screenshot if a card is detected for Y milliseconds
        if found_corners_num >= 3:
            valid_frames += 1
            if valid_frames >= WAIT_FRAMES:
                card = masked_frame[
                    topleft[0] : bottomright[0], topleft[1] : bottomright[1]
                ]
                cv2.imshow("Auto Captured Card", card)
                valid_frames = 0
                cv2.waitKey(0)
                cv2.destroyWindow("Auto Captured Card")
                # cv2.imwrite(f"card-{time.time()}.png", card)
        else:
            valid_frames = 0

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

        # Press ESC or "q" to quit
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord("q"):
            print(f"Shutting Down {app_name}...")
            cam.release()
            return


if __name__ == "__main__":
    main()
