# CardScanner Custom C++ Version (v0.2 Onwards)

This folder contains the custom C++ version of CardScanner. The main logic is in `card_corner_detector.cpp` and `card_corner_detector.h`. The `/houghlines` folder contains the open-source pure C++ implementation of the HoughLines algorithm, referenced from [this repo](https://github.com/frotms/line_detector/tree/master/houghlines).

In `main.cpp`, we use OpenCV to read the image and instantiate the `CardCornerDetector` class to detect the card corners. This external C++ file is only for testing purpose because it is much easier to do image reading and pre/post-processing using OpenCV. Hence, make sure OpenCV v4 is installed before trying to compile the `main.cpp` file.

Finally, the `makefile` is used to compile the C++ files. The `makefile` is only for testing purpose and is not used in the actual CardScanner app.

```bash
# Install OpenCV
brew install opencv

# Compile and Run the C++ file
clear && make && ./main
```
