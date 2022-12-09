# CardScanner Custom C++ Version

This folder contains the custom C++ version of CardScanner. The main logic is in `card_corner_detector.cpp` and `card_corner_detector.h`. The `/houghlines` folder contains the open-source pure C++ implementation of the HoughLines algorithm, referenced from [this repo](https://github.com/frotms/line_detector/tree/master/houghlines).

In `main-cv.cpp`, we use OpenCV to read the image and instantiate the `CardCornerDetector` class to detect the card corners. This external C++ file is only for testing purpose because it is much easier to do image reading and pre/post-processing using OpenCV. The final app will not use OpenCV.
