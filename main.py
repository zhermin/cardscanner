"""Simple proof-of-concept automatic card scanner written in Python

Implementation details were taken from the docs by PQPO at:
https://pqpo.me/2018/09/12/android-camera-real-time-scanning/"""

import cv2
import numpy as np

PARAMS = {
    "maxSize": 300,
    "gaussian_blur_radius": (5, 5),  # higher radius = more blur
    "canny_threshold1": 20,
    "canny_threshold2": 50,
    "dilate_structing_element_size": (3, 3),  # larger kernel = thicker lines
    "OTSU_threshold_min": 0,
    "OTSU_threshold_max": 255,
    "houghlines_threshold": 130,
    "houghlines_min_line_length": 80,
    "houghlines_max_line_gap": 10,
    "detection_ratio": 0.1,
    "min_length_ratio": 0.8,
    "angle_threshold": 5,  # in degrees
}


def process_image(img: np.ndarray) -> np.ndarray:
    # Crop and scale down the image
    img = img[95 : 95 + 530, 225 : 225 + 830]  # coordinates tentatively hardcoded
    img = cv2.resize(
        img, (PARAMS["maxSize"], int(PARAMS["maxSize"] / img.shape[1] * img.shape[0]))
    )

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


def check_lines(lines: np.ndarray, min_length: int, vertical: bool) -> bool:
    for line in lines:
        for x1, y1, x2, y2 in line:
            width, height = abs(x2 - x1), abs(y2 - y1)

            dist = width**2 + height**2
            if dist < min_length**2:
                continue

            if x1 == x2:
                return True

            angle = np.arctan2(height, width)  # * 180 / np.pi
            if vertical:
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
    img: cv2.Mat = process_image(frame)

    # Section off the image into 4 areas to check for lines
    imgH, imgW = img.shape
    detectionH, detectionW = int(imgH * PARAMS["detection_ratio"]), int(
        imgW * PARAMS["detection_ratio"]
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

    # Draw the found lines on the image
    blank_img = np.zeros((imgH, imgW, 3), np.uint8)

    # Print the coordinates of the found lines
    # print(f"Left Lines: {lines_left}")
    # print(f"Right Lines: {lines_right}")
    # print(f"Top Lines: {lines_top}")
    # print(f"Bottom Lines: {lines_bottom}")

    # Draw the found lines on a blank image to check if correctly found
    for line in lines_left:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for line in lines_right:
        for x1, y1, x2, y2 in line:
            cv2.line(
                blank_img,
                (imgW - detectionW + x1, y1),
                (imgW - detectionW + x2, y2),
                (255, 0, 0),
                2,
            )
    for line in lines_top:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for line in lines_bottom:
        for x1, y1, x2, y2 in line:
            cv2.line(
                blank_img,
                (x1, imgH - detectionH + y1),
                (x2, imgH - detectionH + y2),
                (255, 0, 0),
                2,
            )

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
    cv2.imshow("Detected Lines", blank_img)
    cv2.imshow("Preview", img)

    # Press any key to quit the program
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
