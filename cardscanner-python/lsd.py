"""Line Segment Detector in Python OpenCV"""

import cv2
import numpy as np
import sys
from src.Camera import Camera

PARAMS = {
    "max_size": 300,  # width of scaled down image for faster processing
    "frame_scaling_factor": 1,  # ratio of unmasked area to the entire frame (180)
    "mask_aspect_ratio_height": 54,  # CR80 standard card height is 54mm
    "mask_aspect_ratio_width": 86,  # CR80 standard card width is 86mm
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
SHOW_PREVIEW = True


def find_corners(img: np.ndarray) -> tuple[np.ndarray]:
    """Find the corners in the binarized image"""

    imgH, imgW = img.shape
    detectionH = int(imgH * PARAMS["area_detection_ratio"] / 2)
    detectionW = int(imgW * PARAMS["area_detection_ratio"] / 2)
    shift_x, shift_y = imgW - detectionW, imgH - detectionH

    # Section off the image into 4 areas to check for lines
    region_left = img[:, :detectionW]
    region_right = img[:, shift_x:]
    region_top = img[:detectionH, :]
    region_bottom = img[shift_y:, :]

    # Draw the found lines on the preview image and for corner detection
    highlighted_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    preview_regions = np.zeros((imgH, imgW, 3), np.uint8)
    preview_regions[:, :detectionW] = cv2.cvtColor(region_left, cv2.COLOR_GRAY2BGR)
    preview_regions[:, shift_x:] = cv2.cvtColor(region_right, cv2.COLOR_GRAY2BGR)
    preview_regions[:detectionH, :] = cv2.cvtColor(region_top, cv2.COLOR_GRAY2BGR)
    preview_regions[shift_y:, :] = cv2.cvtColor(region_bottom, cv2.COLOR_GRAY2BGR)

    # Section off the image into the 4 corner areas to check for card corners
    region_topleft = img[:detectionH, :detectionW]
    region_topright = img[:detectionH, shift_x:]
    region_bottomleft = img[shift_y:, :detectionW]
    region_bottomright = img[shift_y:, shift_x:]

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


def main():
    # Initialize the camera or image file based on the command line arguments
    use_cam = False
    if len(sys.argv) > 1:
        cam = Camera(int(sys.argv[1]), prevent_flip=True)
        use_cam = True

    while True:
        if use_cam:
            frame_got, frame = cam.get_frame()
            if not frame_got:
                print("No frame to process")
                return
        else:
            filename = "../cardscanner-custom/assets/card.png"
            frame = cv2.imread(filename)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop image to fit the aspect ratio of the card
        imgH, imgW = img.shape
        mask_aspect_ratio = (
            PARAMS["mask_aspect_ratio_height"] / PARAMS["mask_aspect_ratio_width"]
        )
        if use_cam:
            if imgH / imgW > mask_aspect_ratio:
                # crop top and bottom if the image is too tall
                cropH = int((imgH - imgW * mask_aspect_ratio) / 2)
                img = img[cropH : imgH - cropH, :]
            else:
                # crop left and right if the image is too wide
                cropW = int((imgW - imgH / mask_aspect_ratio) / 2)
                img = img[:, cropW : imgW - cropW]

        # Use LSD to detect lines
        img = cv2.GaussianBlur(
            img, (PARAMS["gaussian_blur_radius"], PARAMS["gaussian_blur_radius"]), 0
        )

        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(img)[0]

        # Draw lines on new blank image
        drawn_img = np.zeros(img.shape, dtype=np.uint8)
        if lines is not None and lines.any():
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                cv2.line(drawn_img, (x1, y1), (x2, y2), 255, 1)

        # Find corners
        highlighted_lines, preview_regions, corners = find_corners(drawn_img)

        # Check if the corners are found
        corner_topleft, corner_topright, corner_bottomleft, corner_bottomright = corners
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

        # Card detected if at least 3 corners are found
        if found_corners_num >= 3:
            print("Card Detected!")
        else:
            print("---")

        # Show image
        cv2.imshow("Corners", highlighted_lines)
        cv2.imshow("Original", img)

        # Press ESC or "q" to quit
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord("q"):
            print(f"Shutting Down...")
            if use_cam:
                cam.release()
            return


if __name__ == "__main__":
    main()
