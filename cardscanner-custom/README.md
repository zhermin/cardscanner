# CardScanner Custom C++ Version

Custom C++ Version to perform edge/corner detection on cards to replace the OpenCV library due to extremely limited memory allotment on mobile devices.

# Introduction

This TD showcases the next version v0.2 of the card edge/corner detector. The main improvement over version v0.1 is the removal of all OpenCV dependencies and instead using an open-sourced custom implementation of the line detection algorithm in C++.

The primary goal of removing the dependency on OpenCV is to reduce the eventual app size. As elaborated in the previous TD for v0.1, the OpenCV library takes up a huge amount of storage. Even the mobile bundle requires upwards of 10MB, which severely exceeds the current set limit of 3MB for the entire SDK, including the other models in the auto-scanning pipeline.

However, the performance of the model might worsen in this version because some OpenCV functions previously used were harder to find custom implementations for, such as the Shi-Tomasi corner detection algorithm. Instead, tentatively, a simpler corner detection method is used in this version.

# v0.2 Pure C++ Model (w/o OpenCV)

![v0.2 Model Development](./assets/docs/v0.2%20Model%20Development.png)

1. Image masking, cropping and resizing will be done on the app's frontend
2. After the Java preprocessing steps above, the edge detection model will receive a C++ `byteArray` or `unsigned char *` input which has already been grayscaled
3. Use the custom C++ Canny edge detector and Hough Line transform algorithm
    1. Further crop and scale down the frame for faster processing
    2. The algo applies a **Gaussian blur** to reduce noise
    3. The algo then applies a **Canny edge detector** to detect edges (full details of the Canny algorithm can be found in the v0.1 TD)
    4. I believe the algo *does not* apply a **Dilation** to thicken the edges
    5. The algo finally applies the hough line transform to find lines
4. Check for lines
    1. Section off the image into 4 areas: left, right, top and bottom
    2. Keep only the lines at the 4 side regions
5. Check for corners
    1. Section off the image into the 4 corners: top left, top right, bottom left and bottom right
    2. Instead of using a corner detection algorithm like Shi-Tomasi (used in v0.1 OpenCV version), a simple algorithm to check for corners is used
    3. By checking if there are lines starting or ending at the 4 corner regions, we can get roughly the same performance as the Shi-Tomasi algorithm
6. Draw the found lines and corners on a new image for debugging purposes
7. Return the number of corners found; if 3 or 4 corners found, the auto-capture pipeline can move to the Card Type detection, else, it will restart from the beginning of the pipeline

# v0.3 Pure C++ Model (w/ Corner Score)

## Model Improvements

### Enlarging Input Image

- Guides the user to align the card edges exactly to the guided viewfinder
- Downstream neural network models are often trained on card images with almost no excess space around the card
- These NN models will continue to receive the cropped input frame directly from the guided viewfinder
- However, this card detection model will receive a slightly larger cropped frame to perform the edge detections outside of the guided viewfinder
- This is because the edge detection requires some gap to be able to differentiate edges from background
- This will be handled from the SDK Frontend, where cropping and frame inputs will be preprocessed before sending into this model and other NN models

### Blanking Out Center Exclusion Area

- Possible improvement to the processing speed
- For the center exclusion area, change the pixel values to 0 to avoid line/edge detection there
- Runtime did not exhibit any noticeable change
- Runtime fluctuates from 15ms to 20ms with or without this implementation
- However additional challenges would be introduced
  - Since exclusion area is completely skipped, certain lines/edges could be missed by the algorithm
  - More loops will have to be run to preprocess the image, perhaps the reason for no net change in the runtime
- Trying to split the `unsigned char *` frame into 4 separate regions is tedious without an image processing library like OpenCV
  - Additionally, the runtime might also increase due to the overhead of calling the Canny, Hough Transform and other post-processing algorithms 4 times
- Overall, this idea will likely not improve the model and might actually make it slightly worse due to further complications introduced

### Corner Score

- A corner "accuracy" score is implemented to give more granular feedback and to allow the best frame to be picked if multiple frames with 4 corners are found
- Higher scores will be achieved if the user aligns the card as perfectly to the guided viewfinder as possible
- The model detection zone is from the red border to the yellow border; everything inside of the yellow zone is NOT scanned
- The scoring formula is calculated by checking the distance between the coordinates of the found corner against the guide view corner shown in the figure above

### Streamlined Tuneable Parameters

```cpp
struct {
  float resizedWidth =
      480; // new width of sized down image for faster processing
  float detectionAreaRatio =
      0.15; // ratio of the detection area to the image area (30)
  int cannyLowerThreshold =
      60; // rejected if pixel gradient is below lower threshold
  int cannyUpperThreshold =
      180; // accepted if pixel gradient is above upper threshold
  int houghlineThreshold = 30; // minimum intersections to detect a line
  float houghlineMinLineLengthRatio =
      0.1; // minimum length of a line to detect (30)
  float houghlineMaxLineGapRatio =
      0.1; // maximum gap between two potential lines to join into 1 line (30)
} PARAMS;
```

# Metrics

- Size: Total size of the model and if including imported libraries, if any
- Speed: Time to process each frame
- Accuracy: Auto-capturing when it is supposed to ("Predicted" Positives)
- Stability: NOT auto-capturing when it is not supposed to ("Predicted" Negatives)

|      |                 |         |          |           |
| ---- | --------------- | ------- | -------- | --------- |
|      | Pure Model Size | Speed   | Accuracy | Stability |
| v0.1 | 17 MB w/ OpenCV |         |          |           |
| v0.2 | ~30KB           | 15-20ms |          |           |
| v0.3 | ~30KB           | 15-20ms |          |           |
| v0.4 |                 |         |          |           |
