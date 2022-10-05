import cv2
import numpy as np
import sys

width = 860
height = 540


def main():
    """Straighten card with OpenCV to 860x540 given the 4 corners of the card"""
    # Get the card image file and the 4 corners from the command line
    if len(sys.argv) < 6:
        print(
            "[USAGE] python warp.py <image file> <top left x> <top left y> <top right x> <top right y> <bottom right x> <bottom right y> <bottom left x> <bottom left y>"
        )
        return

    # Read image
    filename = sys.argv[1]
    img = cv2.imread(filename)

    # Get the 4 corners
    corners = np.float32(
        [
            [int(sys.argv[2]), int(sys.argv[3])],
            [int(sys.argv[4]), int(sys.argv[5])],
            [int(sys.argv[6]), int(sys.argv[7])],
            [int(sys.argv[8]), int(sys.argv[9])],
        ]
    )
    print("Corners: ", corners)

    # Sort the corners so that they are in the correct order
    corners_sum = np.sum(corners, axis=1)
    corners_diff = np.diff(corners, axis=1)

    top_left = corners[np.argmin(corners_sum)]
    bottom_right = corners[np.argmax(corners_sum)]
    top_right = corners[np.argmin(corners_diff)]
    bottom_left = corners[np.argmax(corners_diff)]

    # Create a new array with the 4 corners in the correct order
    corners = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype="float32"
    )

    # Get the destination points
    dst = np.float32(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ]
    )

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst)

    # Warp the image
    warped = cv2.warpPerspective(img, M, (width, height))

    # Save image
    filename = filename.split("/")[-1].split(".")[0]
    outfile = rf"../../assets/ktp/{filename}.jpg"
    cv2.imwrite(outfile, warped)

    print(f"[DONE] {outfile}")


if __name__ == "__main__":
    main()
