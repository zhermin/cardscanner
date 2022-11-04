"""Real-time card scanner using Python and OpenCV"""

# External Libraries
import time
import cv2
import numpy as np
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
    "houghlineThreshold": 80,  # minimum intersections to detect a line
    "houghlineMinLineLengthRatio": 0.1,  # minimum length of a line to detect (30)
    "houghlineMaxLineGapRatio": 0.3,  # maximum gap between two potential lines to join into 1 line (30)
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
        "Gaussian Blur Sigma",
        "Parameters",
        int(PARAMS["sigma"] * 100),
        100,
        lambda x: print(f"Gaussian Blur Sigma: {x}"),
    )
    cv2.createTrackbar(
        "Canny Lower Threshold Ratio",
        "Parameters",
        int(PARAMS["cannyLowerThreshold"] * 100),
        100,
        lambda x: print(f"Canny Lower Threshold Ratio: {x / 100}"),
    )
    cv2.createTrackbar(
        "Canny Upper Threshold Ratio",
        "Parameters",
        int(PARAMS["cannyUpperThreshold"] * 100),
        100,
        lambda x: print(f"Canny Upper Threshold Ratio: {x / 100}"),
    )
    cv2.createTrackbar(
        "Houghline Threshold Ratio",
        "Parameters",
        int(PARAMS["houghlineThreshold"] * 100),
        100,
        lambda x: print(f"Houghline Threshold Ratio: {x / 100}"),
    )
    cv2.createTrackbar(
        "Area Detection Ratio",
        "Parameters",
        int(PARAMS["detectionAreaRatio"] * 100),
        100,
        lambda x: print(f"Area Detection Ratio: {x / 100}"),
    )


def update_params() -> None:
    PARAMS["sigma"] = cv2.getTrackbarPos("Gaussian Blur Sigma", "Parameters") / 100
    PARAMS["cannyLowerThreshold"] = (
        cv2.getTrackbarPos("Canny Lower Threshold Ratio", "Parameters") / 100
    )
    PARAMS["cannyUpperThreshold"] = (
        cv2.getTrackbarPos("Canny Upper Threshold Ratio", "Parameters") / 100
    )
    PARAMS["dilate_kernel_size"] = cv2.getTrackbarPos(
        "Dilate Kernel Size", "Parameters"
    )
    PARAMS["houghlineThreshold"] = (
        cv2.getTrackbarPos("Houghline Threshold Ratio", "Parameters") / 100
    )
    PARAMS["detectionAreaRatio"] = (
        cv2.getTrackbarPos("Area Detection Ratio", "Parameters") / 100
    )


def mask_video(img: np.ndarray, camW: int, camH: int) -> np.ndarray:
    """Apply a rectangle mask to capture only a portion of the image"""

    mask_border = (
        int(PARAMS["resizedWidth"] * PARAMS["detectionAreaRatio"]) if DEBUG else 0
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
            PARAMS["resizedWidth"],
            int(PARAMS["resizedWidth"] / img.shape[1] * img.shape[0]),
        ),
    )

    return img, masked_frame, (maskH, maskW, mask_border)


def process_image(img: np.ndarray) -> np.ndarray:
    """Use Canny edge detector to highlight all the lines in the image"""

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preview_grayscale = img.copy()
    if SHOW_PREVIEW:
        preview_grayscale = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(
        img, ksize=(0, 0), sigmaX=PARAMS["sigma"], sigmaY=PARAMS["sigma"]
    )

    # Apply Canny edge detection to find edges
    img = cv2.Canny(
        img,
        int(PARAMS["resizedWidth"] * PARAMS["cannyLowerThreshold"]),
        int(PARAMS["resizedWidth"] * PARAMS["cannyUpperThreshold"]),
    )
    if SHOW_PREVIEW:
        preview_findlines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Dilate the image to strengthen lines and edges
    # img = cv2.dilate(
    #     img,
    #     cv2.getStructuringElement(
    #         cv2.MORPH_RECT,
    #         (
    #             PARAMS["dilate_kernel_size"],
    #             PARAMS["dilate_kernel_size"],
    #         ),
    #     ),
    # )
    # if SHOW_PREVIEW:
    #     preview_end = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return (
        img,
        np.vstack([preview_grayscale, preview_findlines]) if SHOW_PREVIEW else None,
        preview_grayscale,
    )


def find_lines_and_corners(
    img: np.ndarray, mask_dims: tuple[int] = (-1, -1)
) -> tuple[np.ndarray]:
    """Find the lines in the binarized image"""

    imgH, imgW = img.shape
    # maskH, maskW, mask_border = mask_dims
    detectionH = int(imgH * PARAMS["detectionAreaRatio"] / 2)
    detectionW = int(imgW * PARAMS["detectionAreaRatio"] / 2)
    # if DEBUG:
    #     detectionH = int(imgH * mask_border / maskH)
    #     detectionW = int(imgW * mask_border / maskW)
    shift_x, shift_y = imgW - detectionW, imgH - detectionH

    # Section off the image into 4 areas to check for lines
    # TODO: can shift here before houghlines
    region_left = img[:, :detectionW]
    region_right = img[:, shift_x:]
    region_top = img[:detectionH, :]
    region_bottom = img[shift_y:, :]

    # Use Hough Lines Transform to find lines in the image
    def get_houghlines(region: np.ndarray) -> np.ndarray:
        """Use the less efficient Hough Lines Transform to find lines in the image
        HoughLinesP() is faster but returns Cartesian instead of polar coordinates
        Because many lines will be found, return only 5 of them to reduce processing time"""
        # lines = cv2.HoughLines(
        #     region,
        #     1,
        #     np.pi / 180,
        #     int(PARAMS["resizedWidth"] * PARAMS["houghlineThreshold"]),
        # )
        # return lines[:1] if lines is not None else np.array([])

        lines = cv2.HoughLinesP(
            region,
            1,
            np.pi / 180,
            int(PARAMS["resizedWidth"] * PARAMS["houghlineThreshold"]),
            np.array([]),
            int(PARAMS["resizedWidth"] * PARAMS["houghlineMinLineLengthRatio"]),
            int(PARAMS["resizedWidth"] * PARAMS["houghlineMaxLineGapRatio"]),
        )

        return lines if lines is not None else np.array([])

    lines_left, lines_right, lines_top, lines_bottom = map(
        get_houghlines, [region_left, region_right, region_top, region_bottom]
    )

    # Shift the polar coordinate lines to the correct position using cartesian shift values
    highlighted_lines = np.zeros((imgH, imgW, 3), np.uint8)
    preview_regions = np.zeros((imgH, imgW, 3), np.uint8)

    def shift_lines(
        lines: np.ndarray, shift_x: int = 0, shift_y: int = 0
    ) -> np.ndarray:
        if not lines.any():
            return lines
        lines_new = []
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)) + shift_x, int(y0 + 1000 * a) + shift_y
            x2, y2 = int(x0 - 1000 * (-b)) + shift_x, int(y0 - 1000 * a) + shift_y
            lines_new.append(np.array([x1, y1, x2, y2]))
            cv2.line(highlighted_lines, (x1, y1), (x2, y2), RED, 1)
            cv2.line(preview_regions, (x1, y1), (x2, y2), RED, 2)
        return np.array(lines_new)

    lines_left = shift_lines(lines_left)
    lines_right = shift_lines(lines_right, shift_x=shift_x)
    lines_top = shift_lines(lines_top)
    lines_bottom = shift_lines(lines_bottom, shift_y=shift_y)

    # Draw the found lines on the preview image and for corner detection
    preview_regions[:, :detectionW] = cv2.cvtColor(region_left, cv2.COLOR_GRAY2BGR)
    preview_regions[:, shift_x:] = cv2.cvtColor(region_right, cv2.COLOR_GRAY2BGR)
    preview_regions[:detectionH, :] = cv2.cvtColor(region_top, cv2.COLOR_GRAY2BGR)
    preview_regions[shift_y:, :] = cv2.cvtColor(region_bottom, cv2.COLOR_GRAY2BGR)

    def draw_lines(lines: np.ndarray, shift_x: int = 0, shift_y: int = 0) -> None:
        if not lines.any():
            return
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b) + shift_x), int(y0 + 1000 * (a) + shift_y)
            x2, y2 = int(x0 - 1000 * (-b) + shift_x), int(y0 - 1000 * (a) + shift_y)
            # for x1, y1, x2, y2 in line:
            cv2.line(highlighted_lines, (x1, y1), (x2, y2), RED, 1)
            cv2.line(preview_regions, (x1, y1), (x2, y2), RED, 2)

    # draw_lines(lines_left)
    # draw_lines(lines_right, shift_x=shift_x)
    # draw_lines(lines_top)
    # draw_lines(lines_bottom, shift_y=shift_y)

    def intersection(line1: np.ndarray, line2: np.ndarray) -> tuple[float, float]:
        """Find the intersection of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return 0, 0
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return x, y

    def intersection2(line1: np.ndarray, line2: np.ndarray) -> list[int]:
        """Finds the intersection of two lines given in Hesse normal form"""

        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        a, b = np.cos(theta1), np.sin(theta1)
        c, d = np.cos(theta2), np.sin(theta2)
        g = a * d - b * c
        if g == 0:  # if lines are parallel, there will be no intersection
            return None
        x0, y0 = (d * rho1 - b * rho2) / g, (-c * rho1 + a * rho2) / g
        # x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]

    def find_corners(lines1: np.ndarray, lines2: np.ndarray) -> np.ndarray:
        """Find the corners, which are the intersections between two sides' lines
        If multiple corners are found, return the average coordinates of the corners"""
        # lines1, lines2 = all_lines
        if not lines1.any() or not lines2.any():
            return np.array([])

        # corners = []
        for line1 in lines1:
            for line2 in lines2:
                corner = intersection(line1, line2)
                print(line1, line2, corner)
                if (
                    corner is not None
                    and 0 <= corner[0] < imgW
                    and 0 <= corner[1] < imgH
                ):
                    return np.array(corner)  # .append(corner)
        return np.array([])  # np.array(corners) if corners else np.array([])

    # Find the corners of the image
    corner_topleft = find_corners(lines_top, lines_left)
    corner_topright = find_corners(lines_top, lines_right)
    corner_bottomleft = find_corners(lines_bottom, lines_left)
    corner_bottomright = find_corners(lines_bottom, lines_right)

    # corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = map(
    #     find_corners,
    #     [
    #         (lines_top, lines_left),
    #         (lines_top, lines_right),
    #         (lines_bottom, lines_left),
    #         (lines_bottom, lines_right),
    #     ],
    # )

    # Section off the image into the 4 corner areas to check for card corners
    # corner_frame = cv2.cvtColor(highlighted_lines, cv2.COLOR_BGR2GRAY)
    # region_topleft = corner_frame[:detectionH, :detectionW]
    # region_topright = corner_frame[:detectionH, shift_x:]
    # region_bottomleft = corner_frame[shift_y:, :detectionW]
    # region_bottomright = corner_frame[shift_y:, shift_x:]

    # Use Shi-Tomasi corner detection to find corners in the image
    # get_corners = lambda region: cv2.goodFeaturesToTrack(
    #     region, 1, PARAMS["corner_quality_ratio"], 20
    # )

    # corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = map(
    #     get_corners,
    #     [region_topleft, region_topright, region_bottomleft, region_bottomright],
    # )

    # Draw the found lines and corners on a small preview window
    if corner_topleft.any():
        corner_topleft = corner_topleft.astype(int)
        print("topleft", corner_topleft)
    if corner_topright.any():
        corner_topright = corner_topright.astype(int)
        # corner_topright[0] += shift_x
        print("topright", corner_topright)
    if corner_bottomleft.any():
        corner_bottomleft = corner_bottomleft.astype(int)
        # corner_bottomleft[1] += shift_y
        print("bottomleft", corner_bottomleft)
    if corner_bottomright.any():
        corner_bottomright = corner_bottomright.astype(int)
        # corner_bottomright[0] += shift_x
        # corner_bottomright[1] += shift_y
        print("bottomright", corner_bottomright)

    if SHOW_PREVIEW:
        if corner_topleft.any():
            x, y = corner_topleft.astype(int)  # .ravel()
            cv2.circle(highlighted_lines, (x, y), 3, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 3, GREEN, -1)
        if corner_topright.any():
            x, y = corner_topright.astype(int)  # .ravel()
            # x += shift_x
            cv2.circle(highlighted_lines, (x, y), 3, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 3, GREEN, -1)
        if corner_bottomleft.any():
            x, y = corner_bottomleft.astype(int)  # .ravel()
            # y += shift_y
            cv2.circle(highlighted_lines, (x, y), 3, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 3, GREEN, -1)
        if corner_bottomright.any():
            x, y = corner_bottomright.astype(int)  # .ravel()
            # x += shift_x
            # y += shift_y
            cv2.circle(highlighted_lines, (x, y), 3, GREEN, -1)
            cv2.circle(preview_regions, (x, y), 3, GREEN, -1)
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

        # Crop image to be 480 x 301 from the center
        frameW, frameH = 480, 301
        guideW, guideH = 440, 277

        cropped = frame[
            camH // 2 - frameH // 2 : camH // 2 + frameH // 2,
            camW // 2 - frameW // 2 : camW // 2 + frameW // 2,
        ]

        # Process image and get the corners
        img_resized, masked_frame, mask_dims = mask_video(frame, camH, camW)
        img, preview_processed, preview_grayscale = process_image(img_resized)
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

        # Image warping and perspective transform
        warped = img_resized.copy()
        warpH, warpW = warped.shape[:2]
        if found_corners_num == 4:
            dst = np.array(
                [
                    [0, 0],
                    [warpW, 0],
                    [0, warpH],
                    [warpW, warpH],
                ],
                dtype=np.float32,
            )
            src = np.array(
                [
                    corner_topleft,
                    corner_topright,
                    corner_bottomleft,
                    corner_bottomright,
                ],
                dtype=np.float32,
            )
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(warped, M, (warpW, warpH))

        # Corner coordinates
        topleft = int(camW / 2 - mask_dims[0] / 2), int(camH / 2 - mask_dims[1] / 2)
        bottomright = int(camW / 2 + mask_dims[0] / 2), int(camH / 2 + mask_dims[1] / 2)

        # Save card screenshot region first
        card = masked_frame.copy()[
            topleft[0] : bottomright[0], topleft[1] : bottomright[1]
        ]

        # Draw the user feedback
        masked_frame = draw_user_feedback(
            masked_frame, found_corners, topleft, bottomright
        )

        # Show the previews with the edges highlighted
        if SHOW_PREVIEW:
            preview = np.vstack(
                [preview_processed, preview_lines, preview_regions, warped]
            )
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
        if found_corners_num >= 4:
            valid_frames += 1
            if valid_frames >= PARAMS["wait_frames"]:
                valid_frames = 0
                # cv2.imshow("Auto Captured Card", card)
                key = cv2.waitKey(0)
                if key == ord("s"):
                    cv2.imwrite(f"card-{time.time()}.png", card)
                else:
                    cv2.destroyWindow("Auto Captured Card")
        else:
            valid_frames = 0

        # Press "s" to save a screenshot, ESC or "q" to quit
        key = cv2.waitKey(1)
        if key == ord("s"):
            cv2.imwrite(f"../cardscanner-custom/assets/card1.png", card)
        elif key == 27 or key == ord("q"):
            print(f"Shutting Down {app_name}...")
            cam.release()
            return


if __name__ == "__main__":
    main()
