"""Simple proof-of-concept automatic card scanner written in Python

Implementation details were taken from the docs by PQPO at:
https://pqpo.me/2018/09/12/android-camera-real-time-scanning/"""

# External Libraries
import cv2
import numpy as np
from argparse import ArgumentParser

PARAMS = {
    "mask_alpha": 0.75,
    "mask_aspect_ratio": (16, 10),
    "frame_scaling_factor": 0.8,  # ratio of unmasked area to the entire frame
    "max_size": 300,  # scaled down image for faster processing
    "gaussian_blur_radius": (5, 5),  # higher radius = more blur
    "canny_threshold1": 20,
    "canny_threshold2": 50,
    "dilate_structing_element_size": (3, 3),  # larger kernel = thicker lines
    "OTSU_threshold_min": 0,
    "OTSU_threshold_max": 255,
    "houghlines_threshold": 130,
    "houghlines_min_line_length": 80,
    "houghlines_max_line_gap": 10,
    "area_detection_ratio": 0.2,  # ratio of the detection area to the image area
    "min_length_ratio": 0.7,  # ratio of lines to detect to the image edges
    "angle_threshold": 5,  # in degrees
}


def parse_args():
    parser = ArgumentParser(description="Detect a card in an image")
    parser.add_argument(
        "-file", type=str, default=None, help="Path to the image to process"
    )
    return parser.parse_args()


def mask_video(img: np.ndarray) -> np.ndarray:
    imgH, imgW = img.shape

    maskW, maskH = PARAMS["mask_aspect_ratio"]
    maskW, maskH = (
        int(imgW * PARAMS["frame_scaling_factor"]),
        int(imgW * PARAMS["frame_scaling_factor"] * maskH / maskW),
    )

    mask = np.ones(img.shape, np.uint8)
    mask[
        int(imgH / 2 - maskH / 2) : int(imgH / 2 + maskH / 2),
        int(imgW / 2 - maskW / 2) : int(imgW / 2 + maskW / 2),
    ] = 0
    mask_area = mask.astype(bool)

    masked_frame = img.copy()
    masked_frame[mask_area] = cv2.addWeighted(
        img, 1 - PARAMS["mask_alpha"], mask, PARAMS["mask_alpha"], gamma=0
    )[mask_area]
    img = img[
        int(imgH / 2 - maskH / 2) : int(imgH / 2 + maskH / 2),
        int(imgW / 2 - maskW / 2) : int(imgW / 2 + maskW / 2),
    ]
    img = cv2.resize(
        img, (PARAMS["max_size"], int(PARAMS["max_size"] / img.shape[1] * img.shape[0]))
    )

    cv2.imshow("Masked Frame", masked_frame)
    return img


def process_image(img: np.ndarray) -> np.ndarray:
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
    imgH, imgW = img.shape

    # Section off the image into 4 areas to check for lines
    detectionH, detectionW = int(imgH * PARAMS["area_detection_ratio"]), int(
        imgW * PARAMS["area_detection_ratio"]
    )

    region_left = img[:, :detectionW]
    region_right = img[:, imgW - detectionW :]
    region_top = img[:detectionH, :]
    region_bottom = img[imgH - detectionH :, :]

    # Show the cropped regions used to find the lines
    # cv2.imshow("region_left", region_left)
    # cv2.imshow("region_right", region_right)
    # cv2.imshow("region_top", region_top)
    # cv2.imshow("region_bottom", region_bottom)

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

    # Draw the found lines on a blank image
    if True:
        blank_img = np.zeros((imgH, imgW, 3), np.uint8)

        # Draw the found lines on a blank image to check if correctly found
        if lines_left is not None and lines_left.any():
            for line in lines_left:
                for x1, y1, x2, y2 in line:
                    cv2.line(blank_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if lines_right is not None and lines_right.any():
            for line in lines_right:
                for x1, y1, x2, y2 in line:
                    cv2.line(
                        blank_img,
                        (imgW - detectionW + x1, y1),
                        (imgW - detectionW + x2, y2),
                        (0, 0, 255),
                        2,
                    )
        if lines_top is not None and lines_top.any():
            for line in lines_top:
                for x1, y1, x2, y2 in line:
                    cv2.line(blank_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if lines_bottom is not None and lines_bottom.any():
            for line in lines_bottom:
                for x1, y1, x2, y2 in line:
                    cv2.line(
                        blank_img,
                        (x1, imgH - detectionH + y1),
                        (x2, imgH - detectionH + y2),
                        (0, 0, 255),
                        2,
                    )

        cv2.imshow("Detected Lines", blank_img)

    return lines_left, lines_right, lines_top, lines_bottom


def check_lines(lines: np.ndarray, min_length: int, is_vertical: bool) -> bool:
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

            angle = np.arctan2(height, width)
            if is_vertical:
                if abs(90 - angle) < PARAMS["angle_threshold"]:
                    return True
            else:
                if abs(angle) < PARAMS["angle_threshold"]:
                    return True
    return False


def main() -> None:
    # Read an image and convert to grayscale
    img_file = "./sample-card.jpeg"
    frame = cv2.imread(img_file, 0)

    # Crop and scale down the image
    img: cv2.Mat = mask_video(img)
    img = process_image(frame)
    imgH, imgW = img.shape

    lines_left, lines_right, lines_top, lines_bottom = find_lines(img)

    # Check if the lines are vertical or horizontal enough for a card
    min_length_H = int(imgH * PARAMS["min_length_ratio"])
    min_length_W = int(imgW * PARAMS["min_length_ratio"])

    if (
        check_lines(lines_left, min_length_H, True)
        and check_lines(lines_right, min_length_H, True)
        and check_lines(lines_top, min_length_W, False)
        and check_lines(lines_bottom, min_length_W, False)
    ):
        print("Card Detected!")
    else:
        print("No Card Detected!")

    # Show the final black and white image with the edges highlighted
    cv2.imshow("Edge Detection Preview", img)

    # Press any key to quit the program
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
