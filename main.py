"""Simple proof-of-concept real-time card scanner written in Python

Implementation details were taken from the docs by PQPO at:
https://pqpo.me/2018/09/12/android-camera-real-time-scanning/"""

# External Libraries
import cv2
import numpy as np
from argparse import ArgumentParser
import time
from collections import defaultdict

# Custom Libraries
from Camera import Camera

PARAMS = {
    "max_size": 300,  # scaled down image for faster processing
    "mask_aspect_ratio": (86, 54),  # CR80 standard card size is 86mm x 54mm
    "inner_mask": False,  # shows an inner rectangle mask to the user
    "mask_alpha": 0.8,  # opacity of the mask
    "frame_scaling_factor": 0.6,  # ratio of unmasked area to the entire frame
    "alpha_contrast": 1.5,  # higher value = more contrast (0 to 3)
    "beta_brightness": 0,  # higher value = brighter (-100 to 100)
    "gaussian_blur_radius": 5,  # higher radius = more blur
    "canny_threshold1": 20,
    "canny_threshold2": 50,
    "dilate_structing_element_size": 4,  # larger kernel = thicker lines
    "OTSU_threshold_min": 0,
    "OTSU_threshold_max": 255,
    # Note: Houghlines params are based off of the hard-coded max size of 300
    "houghlines_threshold": 100,  # minimum intersections to detect a line, 130
    "houghlines_min_line_length": 80,  # minimum length of a line, 80
    "houghlines_max_line_gap": 10,  # maximum gap between two points to form a line, 10
    "area_detection_ratio": 0.1,  # ratio of the detection area to the image area
    "corner_quality_ratio": 0.5,  # higher value = stricter corner detection
    "min_length_ratio": 0.9,  # ratio of lines to detect to the image edges
    "angle_threshold": 10,  # in degrees, 5
}

RUNTIMES = defaultdict(list)
DEBUG = False
RED = (0, 0, 255)
GREEN = (0, 255, 0)


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

    t0 = cv2.getTickCount()
    mask_border = (
        int(PARAMS["max_size"] * PARAMS["area_detection_ratio"])
        if PARAMS["inner_mask"]
        else 0
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
        mask_border if PARAMS["inner_mask"] else -1,
    )
    mask_area = mask.astype(bool)
    get_runtime("- Create Mask", t0)

    t0 = cv2.getTickCount()
    masked_frame = img.copy()
    masked_frame[mask_area] = cv2.addWeighted(
        img, 1 - PARAMS["mask_alpha"], mask, PARAMS["mask_alpha"], gamma=0
    )[mask_area]
    get_runtime("- Apply Mask", t0)

    t0 = cv2.getTickCount()
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
    get_runtime("- Crop/Scale Image", t0)

    return img, masked_frame, (maskH, maskW, mask_border)


def process_image(img: np.ndarray) -> np.ndarray:
    """Binarize the image to highlight all the lines"""

    # Increase contrast and brightness
    t0 = cv2.getTickCount()
    img = cv2.convertScaleAbs(
        img, alpha=PARAMS["alpha_contrast"], beta=PARAMS["beta_brightness"]
    )
    get_runtime("- Increase Contrast/Brightness", t0)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image to reduce noise
    img = cv2.GaussianBlur(
        img, (PARAMS["gaussian_blur_radius"], PARAMS["gaussian_blur_radius"]), 0
    )

    # Apply Canny edge detection to the image to find edges
    img = cv2.Canny(img, PARAMS["canny_threshold1"], PARAMS["canny_threshold2"])

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

    # Binarize and threshold the image to remove interference
    _, img = cv2.threshold(
        img,
        PARAMS["OTSU_threshold_min"],
        PARAMS["OTSU_threshold_max"],
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return img


def find_lines_and_corners(
    img: np.ndarray, mask_dims: tuple[int], SHOW_PREVIEW: bool
) -> tuple[np.ndarray]:
    """Find the lines in the binarized image"""

    imgH, imgW = img.shape
    maskH, maskW, mask_border = mask_dims
    if PARAMS["inner_mask"]:
        detectionH = int(imgH * mask_border / maskH)
        detectionW = int(imgW * mask_border / maskW)
    else:
        detectionH = int(imgH * PARAMS["area_detection_ratio"] / 2)
        detectionW = int(imgW * PARAMS["area_detection_ratio"] / 2)

    # Section off the image into 4 areas to check for lines
    region_left = img[:, :detectionW]
    region_right = img[:, imgW - detectionW :]
    region_top = img[:detectionH, :]
    region_bottom = img[imgH - detectionH :, :]

    # Use Probabilistic Hough Transform to find lines in the image
    get_houghlines = lambda region: cv2.HoughLinesP(
        region,
        1,
        np.pi / 180,
        PARAMS["houghlines_threshold"],
        np.array([]),
        PARAMS["houghlines_min_line_length"],
        PARAMS["houghlines_max_line_gap"],
    )

    lines_left, lines_right, lines_top, lines_bottom = map(
        get_houghlines, [region_left, region_right, region_top, region_bottom]
    )

    # Section off the image into the 4 corner areas to check for card corners
    region_topleft = img[:detectionH, :detectionW]
    region_topright = img[:detectionH, imgW - detectionW :]
    region_bottomleft = img[imgH - detectionH :, :detectionW]
    region_bottomright = img[imgH - detectionH :, imgW - detectionW :]

    get_corners = lambda region: cv2.goodFeaturesToTrack(
        region, 1, PARAMS["corner_quality_ratio"], 20
    )

    # Use Shi-Tomasi corner detection to find corners in the image
    corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = map(
        get_corners,
        [region_topleft, region_topright, region_bottomleft, region_bottomright],
    )

    # Draw the found lines and corners on a small preview window
    if SHOW_PREVIEW:
        regions_preview = np.zeros((imgH, imgW, 3), np.uint8)
        regions_preview[:, :detectionW] = cv2.cvtColor(region_left, cv2.COLOR_GRAY2BGR)
        regions_preview[:, imgW - detectionW :] = cv2.cvtColor(
            region_right, cv2.COLOR_GRAY2BGR
        )
        regions_preview[:detectionH, :] = cv2.cvtColor(region_top, cv2.COLOR_GRAY2BGR)
        regions_preview[imgH - detectionH :, :] = cv2.cvtColor(
            region_bottom, cv2.COLOR_GRAY2BGR
        )

        if lines_left is not None and lines_left.any():
            for line in lines_left:
                for x1, y1, x2, y2 in line:
                    cv2.line(regions_preview, (x1, y1), (x2, y2), RED, 2)
        if lines_right is not None and lines_right.any():
            for line in lines_right:
                for x1, y1, x2, y2 in line:
                    cv2.line(
                        regions_preview,
                        (imgW - detectionW + x1, y1),
                        (imgW - detectionW + x2, y2),
                        RED,
                        2,
                    )
        if lines_top is not None and lines_top.any():
            for line in lines_top:
                for x1, y1, x2, y2 in line:
                    cv2.line(regions_preview, (x1, y1), (x2, y2), RED, 2)
        if lines_bottom is not None and lines_bottom.any():
            for line in lines_bottom:
                for x1, y1, x2, y2 in line:
                    cv2.line(
                        regions_preview,
                        (x1, imgH - detectionH + y1),
                        (x2, imgH - detectionH + y2),
                        RED,
                        2,
                    )

        if corner_topleft is not None:
            x, y = corner_topleft.astype(int).ravel()
            cv2.circle(regions_preview, (x, y), 3, GREEN, -1)
        if corner_topright is not None:
            x, y = corner_topright.astype(int).ravel()
            x += imgW - detectionW
            cv2.circle(regions_preview, (x, y), 3, GREEN, -1)
        if corner_bottomleft is not None:
            x, y = corner_bottomleft.astype(int).ravel()
            y += imgH - detectionH
            cv2.circle(regions_preview, (x, y), 3, GREEN, -1)
        if corner_bottomright is not None:
            x, y = corner_bottomright.astype(int).ravel()
            x += imgW - detectionW
            y += imgH - detectionH
            cv2.circle(regions_preview, (x, y), 3, GREEN, -1)
    else:
        regions_preview = None

    return (
        regions_preview,
        [corner_topleft, corner_topright, corner_bottomleft, corner_bottomright],
        [lines_left, lines_right, lines_top, lines_bottom],
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
            # print(
            #     f"{'Vertical' if is_vertical else 'Horizontal'}, Angle: {angle:.2f}deg, Width: {width}, Height: {height}, Min Length: {min_length}"
            # )
            if is_vertical:
                if abs(90 - angle) < PARAMS["angle_threshold"]:
                    return True
            else:
                if abs(angle) < PARAMS["angle_threshold"]:
                    return True
    return False


def draw_text(
    image: np.ndarray,
    label: str,
    coords: tuple[int] = (50, 50),
    color: tuple[int] = (255, 255, 255),
) -> None:
    """Draw text on the image"""
    cv2.putText(
        img=image,
        text=label,
        org=coords,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        color=color,
        thickness=2,
    )


def get_runtime(name: str, t0: float) -> None:
    """Print the runtime of the function in milliseconds"""
    t1 = cv2.getTickCount()
    runtime = (t1 - t0) / cv2.getTickFrequency() * 1000
    RUNTIMES[name].append(runtime)


def main() -> None:
    app_name = "Card Scanner"
    print(f"Initialized {app_name}")

    # Parse command line arguments
    args = parse_args()
    video_src = args.cam if args.cam is not None else args.video
    SHOW_PREVIEW = bool(args.preview)

    # Initialise FPS timings
    prev_frame_time, cur_frame_time = 0, 0

    # Initialise the video capturing object
    t0 = cv2.getTickCount()
    cam = Camera(video_src, prevent_flip=True)
    camH, camW = cam.get_frame_size()
    get_runtime("Initialize Camera", t0)

    while True:
        t0 = cv2.getTickCount()
        frame_got, frame = cam.get_frame()
        if not frame_got:
            print("No frame to process")
            return
        get_runtime("Read Frame", t0)

        # Crop and scale down the image
        t0 = cv2.getTickCount()
        img, masked_frame, mask_dims = mask_video(frame, camH, camW)
        get_runtime("Mask Video", t0)

        t0 = cv2.getTickCount()
        img = process_image(img)
        get_runtime("Process Image", t0)

        t0 = cv2.getTickCount()
        regions_preview, corners, lines = find_lines_and_corners(
            img, mask_dims, SHOW_PREVIEW
        )
        get_runtime("Find Lines", t0)

        lines_left, lines_right, lines_top, lines_bottom = lines
        corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = corners

        # Check if the lines are vertical or horizontal enough for a card
        t0 = cv2.getTickCount()
        min_length_H = int(img.shape[0] * PARAMS["min_length_ratio"])
        min_length_W = int(img.shape[1] * PARAMS["min_length_ratio"])

        draw_line = lambda found_line, x1, y1, x2, y2: cv2.line(
            masked_frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            GREEN if found_line else RED,
            3,
        )

        draw_line(
            check_lines(lines_left, min_length_H, True),
            camH / 2 - mask_dims[1] / 2,
            camW / 2 - mask_dims[0] / 2,
            camH / 2 - mask_dims[1] / 2,
            camW / 2 + mask_dims[0] / 2,
        )
        draw_line(
            check_lines(lines_right, min_length_H, True),
            camH / 2 + mask_dims[1] / 2,
            camW / 2 - mask_dims[0] / 2,
            camH / 2 + mask_dims[1] / 2,
            camW / 2 + mask_dims[0] / 2,
        )
        draw_line(
            check_lines(lines_top, min_length_W, False),
            camH / 2 - mask_dims[1] / 2,
            camW / 2 - mask_dims[0] / 2,
            camH / 2 + mask_dims[1] / 2,
            camW / 2 - mask_dims[0] / 2,
        )
        draw_line(
            check_lines(lines_bottom, min_length_W, False),
            camH / 2 - mask_dims[1] / 2,
            camW / 2 + mask_dims[0] / 2,
            camH / 2 + mask_dims[1] / 2,
            camW / 2 + mask_dims[0] / 2,
        )

        # Check if the corners are found
        draw_circle = lambda found_circle, x, y: cv2.circle(
            masked_frame,
            (int(x), int(y)),
            10,
            GREEN if found_circle is not None else RED,
            -1,
        )

        draw_circle(
            corner_topleft, camH / 2 - mask_dims[1] / 2, camW / 2 - mask_dims[0] / 2
        )
        draw_circle(
            corner_topright, camH / 2 + mask_dims[1] / 2, camW / 2 - mask_dims[0] / 2
        )
        draw_circle(
            corner_bottomleft, camH / 2 - mask_dims[1] / 2, camW / 2 + mask_dims[0] / 2
        )
        draw_circle(
            corner_bottomright, camH / 2 + mask_dims[1] / 2, camW / 2 + mask_dims[0] / 2
        )
        get_runtime("Check Lines and Corners", t0)

        # Calculate FPS
        cur_frame_time = time.perf_counter()
        fps = 1 / (cur_frame_time - prev_frame_time)
        prev_frame_time = cur_frame_time
        RUNTIMES["FPS"].append(fps)

        if DEBUG:
            draw_text(masked_frame, f"FPS: {int(fps)}", coords=(camW - 50, 100))

        # Show the previews with the edges highlighted
        cv2.imshow(app_name, masked_frame)
        if SHOW_PREVIEW:
            cv2.imshow("Detected Edges", regions_preview)

        # Press ESC or "q" to quit
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord("q"):
            print(f"Shutting Down {app_name}...")
            cam.release()
            return


if __name__ == "__main__":
    main()

    if DEBUG:
        print("\nRuntimes:")
        for name, runtime in RUNTIMES.items():
            print(f"{name}")
            print(f"AVG: {np.mean(runtime):.3f}ms")
            print(f"STD: {np.std(runtime):.3f}ms")
            print("---")
