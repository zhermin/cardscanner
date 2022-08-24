"""Simple proof-of-concept automatic card scanner written in Python

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

# Note: Houghlines params are based off of the hard-coded max size of 300
PARAMS = {
    "mask_alpha": 0.75,
    "mask_aspect_ratio": (86, 54),  # CR80 standard card size is 86mm x 54mm
    "frame_scaling_factor": 0.6,  # ratio of unmasked area to the entire frame
    "max_size": 300,  # scaled down image for faster processing
    "gaussian_blur_radius": (5, 5),  # higher radius = more blur
    "canny_threshold1": 20,
    "canny_threshold2": 50,
    "dilate_structing_element_size": (3, 3),  # larger kernel = thicker lines
    "OTSU_threshold_min": 0,
    "OTSU_threshold_max": 255,
    "houghlines_threshold": 100,  # minimum intersections to detect a line, 130
    "houghlines_min_line_length": 50,  # minimum length of a line, 80
    "houghlines_max_line_gap": 50,  # maximum gap between two points to form a line, 10
    "area_detection_ratio": 0.1,  # ratio of the detection area to the image area
    "min_length_ratio": 0.9,  # ratio of lines to detect to the image edges
    "angle_threshold": 10,  # in degrees, 5
}

DRAW_FOUND_LINES = True
RUNTIMES = defaultdict(list)


def parse_args():
    parser = ArgumentParser(description="Detect a card in an image")
    parser.add_argument(
        "-file", type=str, default=None, help="Path to the image to process"
    )
    return parser.parse_args()


def mask_video(img: np.ndarray, camW: int, camH: int) -> np.ndarray:
    """Apply a rectangle mask to capture only a portion of the image"""

    t0 = cv2.getTickCount()
    maskW, maskH = PARAMS["mask_aspect_ratio"]
    maskW, maskH = (
        int(camW * PARAMS["frame_scaling_factor"]),
        int(camW * PARAMS["frame_scaling_factor"] * maskH / maskW),
    )

    mask = np.ones(img.shape, np.uint8)
    mask[
        int(camH / 2 - maskH / 2) : int(camH / 2 + maskH / 2),
        int(camW / 2 - maskW / 2) : int(camW / 2 + maskW / 2),
    ] = 0
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
        img, (PARAMS["max_size"], int(PARAMS["max_size"] / img.shape[1] * img.shape[0]))
    )
    get_runtime("- Crop/Scale Image", t0)

    return img, masked_frame


def process_image(img: np.ndarray) -> np.ndarray:
    """Binarize the image to highlight all the lines"""

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image to remove noise
    img = cv2.GaussianBlur(img, (PARAMS["gaussian_blur_radius"]), 0)

    # Apply Canny edge detection to the image to find edges
    img = cv2.Canny(img, PARAMS["canny_threshold1"], PARAMS["canny_threshold2"])

    # Dilate the image to strengthen lines and edges
    img = cv2.dilate(
        img,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, PARAMS["dilate_structing_element_size"]
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


def find_lines(img: np.ndarray) -> tuple[np.ndarray]:
    """Find the lines in the binarized image"""

    imgH, imgW = img.shape

    # Section off the image into 4 areas to check for lines
    detectionH, detectionW = int(imgH * PARAMS["area_detection_ratio"]), int(
        imgW * PARAMS["area_detection_ratio"]
    )

    region_left = img[:, :detectionW]
    region_right = img[:, imgW - detectionW :]
    region_top = img[:detectionH, :]
    region_bottom = img[imgH - detectionH :, :]

    if DRAW_FOUND_LINES:
        regions_preview = np.zeros((imgH, imgW, 3), np.uint8)
        regions_preview[:, :detectionW] = cv2.cvtColor(region_left, cv2.COLOR_GRAY2BGR)
        regions_preview[:, imgW - detectionW :] = cv2.cvtColor(
            region_right, cv2.COLOR_GRAY2BGR
        )
        regions_preview[:detectionH, :] = cv2.cvtColor(region_top, cv2.COLOR_GRAY2BGR)
        regions_preview[imgH - detectionH :, :] = cv2.cvtColor(
            region_bottom, cv2.COLOR_GRAY2BGR
        )

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

    # Draw the found lines on a blank image to check if correctly found
    if DRAW_FOUND_LINES:
        if lines_left is not None and lines_left.any():
            for line in lines_left:
                for x1, y1, x2, y2 in line:
                    cv2.line(regions_preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if lines_right is not None and lines_right.any():
            for line in lines_right:
                for x1, y1, x2, y2 in line:
                    cv2.line(
                        regions_preview,
                        (imgW - detectionW + x1, y1),
                        (imgW - detectionW + x2, y2),
                        (0, 0, 255),
                        2,
                    )
        if lines_top is not None and lines_top.any():
            for line in lines_top:
                for x1, y1, x2, y2 in line:
                    cv2.line(regions_preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if lines_bottom is not None and lines_bottom.any():
            for line in lines_bottom:
                for x1, y1, x2, y2 in line:
                    cv2.line(
                        regions_preview,
                        (x1, imgH - detectionH + y1),
                        (x2, imgH - detectionH + y2),
                        (0, 0, 255),
                        2,
                    )

    if not DRAW_FOUND_LINES:
        regions_preview = None

    return regions_preview, lines_left, lines_right, lines_top, lines_bottom


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
            print(
                f"{'Vertical' if is_vertical else 'Horizontal'}, Angle: {angle:.2f}deg, Width: {width}, Height: {height}, Min Length: {min_length}"
            )
            if is_vertical:
                if abs(90 - angle) < PARAMS["angle_threshold"]:
                    return True
            else:
                if abs(angle) < PARAMS["angle_threshold"]:
                    return True
    return False


def draw_text(image: np.ndarray, label: str, coords: tuple[int] = (50, 50)) -> None:
    """Draw text on the image"""
    cv2.putText(
        img=image,
        text=label,
        org=coords,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        color=(255, 255, 255),
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

    # Initialise FPS timings
    prev_frame_time, cur_frame_time = 0, 0

    # Initialise the video capturing object
    t0 = cv2.getTickCount()
    cam = Camera(1, prevent_flip=True)
    camH, camW = cam.get_frame_size()
    get_runtime("Initialize Camera", t0)

    while True:
        t0 = cv2.getTickCount()
        frame_got, frame = cam.get_frame()
        if not frame_got:
            print("Failed to get frame from camera")
            break
        get_runtime("Read Frame", t0)

        # Crop and scale down the image
        t0 = cv2.getTickCount()
        img, masked_frame = mask_video(frame, camH, camW)
        get_runtime("Mask Video", t0)

        t0 = cv2.getTickCount()
        img = process_image(img)
        get_runtime("Process Image", t0)

        t0 = cv2.getTickCount()
        regions_preview, lines_left, lines_right, lines_top, lines_bottom = find_lines(
            img
        )
        get_runtime("Find Lines", t0)

        # Check if the lines are vertical or horizontal enough for a card
        t0 = cv2.getTickCount()
        min_length_H = int(img.shape[0] * PARAMS["min_length_ratio"])
        min_length_W = int(img.shape[1] * PARAMS["min_length_ratio"])

        if (
            sum(
                [
                    check_lines(lines_left, min_length_H, True),
                    check_lines(lines_right, min_length_H, True),
                    check_lines(lines_top, min_length_W, False),
                    check_lines(lines_bottom, min_length_W, False),
                ]
            )
            >= 3
        ):
            draw_text(masked_frame, "Card Detected!", coords=(camW - 50, 50))
        else:
            draw_text(masked_frame, "No Card Detected!", coords=(camW - 50, 50))
        get_runtime("Check Lines", t0)

        # Calculate FPS
        cur_frame_time = time.perf_counter()
        fps = 1 / (cur_frame_time - prev_frame_time)
        prev_frame_time = cur_frame_time
        draw_text(masked_frame, f"FPS: {int(fps)}", coords=(camW - 50, 100))

        # Show the previews with the edges highlighted
        cv2.imshow(app_name, masked_frame)
        if DRAW_FOUND_LINES:
            cv2.imshow("Detected Edges", regions_preview)

        # Press esc or "q" to quit
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord("q"):
            print(f"Shutting Down {app_name}...")
            cam.release()
            return


if __name__ == "__main__":
    main()

    print("\nRuntimes:")
    for name, runtime in RUNTIMES.items():
        print(f"{name}")
        print(f"AVG: {np.mean(runtime):.3f}ms")
        print(f"STD: {np.std(runtime):.3f}ms")
        print("---")
