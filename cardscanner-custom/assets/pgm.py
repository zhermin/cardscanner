import cv2
import sys, os


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("[USAGE] python pgm.py <image file> <optional: width> <optional: height>")
        return

    # Read image
    filename = sys.argv[1]
    img = cv2.imread(filename)

    # If width and height are not specified, use the original image size
    if len(sys.argv) == 3:
        width = int(sys.argv[2])
        height = int(img.shape[0] * width / img.shape[1])
    elif len(sys.argv) == 4:
        width = int(sys.argv[2])
        height = int(sys.argv[3])
    else:
        height, width, channels = img.shape

    # Resize and convert to grayscale
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check if filename already exists and append number if so
    i = 1
    while True:
        new_filename = f"{filename.split('.')[0]}{i}.pgm"
        if not os.path.isfile(new_filename):
            break
        i += 1

    # Save image as portable grayscale bitmap
    cv2.imwrite(new_filename, img)

    print(f"[DONE] {new_filename}")


if __name__ == "__main__":
    main()
