# [TD] Identity Card Edge/Corner Detection

`Algorithm Engineer: Zac Tam Zher Min`\
`Supervisor: Yan Xuyan`

<details open>
<summary>Table of Contents</summary>

- [Introduction](#introduction)
- [Requirement](#requirement)
- [General Use-Case Demo](#general-use-case-demo)
- [Scenarios to be Accepted or Rejected by the Model](#scenarios-to-be-accepted-or-rejected-by-the-model)
- [v0.1 OpenCV Model](#v01-opencv-model)
  - [Analysis](#analysis)
    - [Pros](#pros)
    - [Cons](#cons)
    - [Major Con: The OpenCV Library](#major-con-the-opencv-library)
      - [Currently: Use OpenCV as is](#currently-use-opencv-as-is)
      - [Replace OpenCV functions with custom implementation](#replace-opencv-functions-with-custom-implementation)
      - [Train a small NN model to classify a valid image (3 corners detected)](#train-a-small-nn-model-to-classify-a-valid-image-3-corners-detected)
- [v0.2 Pure C++ Model (w/o OpenCV)](#v02-pure-c-model-wo-opencv)
- [v0.3 Enlarged Input \& Corner Scores](#v03-enlarged-input--corner-scores)
  - [Model Improvements](#model-improvements)
    - [Enlarging Input Image](#enlarging-input-image)
    - [Blanking Out Center Exclusion Area](#blanking-out-center-exclusion-area)
    - [Corner Scores](#corner-scores)
- [v0.5 Strictly 3 Corners](#v05-strictly-3-corners)
- [v0.6 Moving Average Corners](#v06-moving-average-corners)
- [Final Overall Model Architecture](#final-overall-model-architecture)
- [Final Tuneable Parameters](#final-tuneable-parameters)
- [Model Performance Evaluation](#model-performance-evaluation)
- [Sources](#sources)
  - [Competitor Research](#competitor-research)
    - [Overview of this SDK](#overview-of-this-sdk)

</details>

# Introduction

This TD showcases a simple real-time card detector using non-deep learning, purely image processing OpenCV solution based on an open-source Android implementation on GitHub by using edge and corner detection to identify "cards". The goal is to then convert from the Python POC to a C++ API to be implemented in an APK, whereby the automatically captured image of the detected card will be passed to the e-KTP card type, flip and quality checks.

The whole model development processes are shown as below, taking an e-KTP card as an example:

1. Use some traditional image processing methods to detect environment light condition, **if there is enough light, go to the next step**
2. Use Edge Detection model to capture the card edge, **if "at least 3 corners of the card are visible within the guided viewfinder", then image is captured, and send the image (the whole card with almost no background information) will go to the next step**
3. Develop card type check model, **if confirm the card is KTP, then go to the next step**
4. Develop card flip check model, **if confirm the card is in right direction, then go to the next step**
5. Develop card quality check model, **if confirm the card is not blurry, block..., then go to the next step**

This TD aims to handle the **Edge Detection** step amidst the entire e-KTP Auto Scanning pipeline shown above.

# Requirement

The target of this IC Edge Detection model is to allow **"at least 3 corners of the card are visible within the guided viewfinder"** before the process moves to the next step, "Card Type Check".

# General Use-Case Demo

![General Use-Case Demo](assets/demo/demo-compressed.mov)

# Scenarios to be Accepted or Rejected by the Model

**Note:** Due to the low resolution of the webcam and low quality of the printed KTP card used in the video demos below, the words on the sample KTP card will not be very legible

| Scenario | Condition(s) | Expected Handling (Auto-Capture Decision) | Demo Video | Remarks |
| -------- | ------------ | ----------------------------------------- | ---------- | ------- |
| Happy Flow | All 5 critical requirements are met | Auto-capture should be triggered as it fulfils all 5 critical requirements:<br><br>1. Card is ID e-KTP; AND<br>2. Card is generally free of quality issue (not blurry, not blocked); AND<br>3. At least 3 corners of the card are visible within the guided viewfinder; AND<br>4. Well-lit condition (good lighting; not too dark, not too bright that it diminishes card details); AND<br>5. Card position is in a proper angle<br><br>Auto-capture should be triggered | ![Scenario 1](assets/demo/scenario1-happyflow.mov) |     |
| At least 3 corners of the card are visible within the guided viewfinder | Only 1 corner blocked | Auto-capture should be triggered as it fulfils all 5 critical requirements | ![Scenario 2](assets/demo/scenario2-blocked1.mov) | The model will auto-capture even if the object is blocking the center of the card with the key information as long as it detects 3 corners |
| Less than 3 corners of the card are visible within the guided viewfinder | 2 corners blocked by object | Auto-capture should not be triggered as "at least 3 corners of the card is visible" is a critical requirement | ![Scenario 3](assets/demo/scenario3-blocked2.mov) |     |
|     | 3 corners blocked by object | Auto-capture should not be triggered as "at least 3 corners of the card is visible" is a critical requirement | ![Scenario 4](assets/demo/scenario4-blocked3.mov) |     |
|     | Card is not horizontally aligned to the guided viewfinder | Auto-capture should not be triggered as "at least 3 corners of the card is visible" is a critical requirement | ![Scenario 5](assets/demo/scenario5-horizontal.mov) |     |
|     | Card is not vertically aligned to the guided viewfinder | Auto-capture should not be triggered as "at least 3 corners of the card is visible" is a critical requirement | ![Scenario 6](assets/demo/scenario6-vertical.mov) |     |
|     | Card is too zoomed out, skewed or slanted away from the guided viewfinder | Auto-capture should not be triggered as "at least 3 corners of the card is visible" is a critical requirement | ![Scenario 7](assets/demo/scenario7-skew.mov) |     |
| **Edge Scenario** | **Condition(s)** | **Expected Handling (Auto-Capture Decision)** | **Demo Video** | **Remarks** |
| Noisy backgrounds | Example: If holding the card in hand or on a background with many lines or dots | Auto-capture should be triggered as it fulfils all 5 critical requirements | ![Scenario 8](assets/demo/scenario8-hand.mov) | This may not always auto-capture and some adjustments by the user may have to be made as the model may face difficulty in such scenarios especially if the user does not fully align the card in the guided viewfinder |
| Similar coloured backgrounds | In the case of the KTP, a similar background colour would be blue and light backgrounds | Auto-capture should be triggered as it fulfils all 5 critical requirements | ![Scenario 9](assets/demo/scenario9-white.mov) | This may not always auto-capture and the user might have to position the card in front of a different background or try to hold it in hand |
| Low exposure | General poor lighting conditions | Auto-capture should not be triggered as "well-lit condition (good lighting; not too dark, not too bright that it diminishes card details)" is a critical requirement | ![Scenario 10](assets/demo/scenario10-lowexposure.mov) | The model will still auto-capture if it can detect the 3 corners because it does not take into account the lighting conditions although the user should reposition to a better lighting condition |
| Strong shadows blocking card features, resulting in poor lighting conditions | Shadows cast by the user or objects in the way of the light source | Auto-capture should not be triggered as "well-lit condition (good lighting; not too dark, not too bright that it diminishes card details)" is a critical requirement | ![Scenario 11](assets/demo/scenario11-shadow.mov) | As above |
| High exposure blocking card features | Direct exposure to light source | Auto-capture should not be triggered as "well-lit condition (good lighting; not too dark, not too bright that it diminishes card details)" is a critical requirement | ![Scenario 12](assets/demo/scenario12-highexposure.mov) | As above |
| Glare blocking card features | Direct exposure to light source | Auto-capture should not be triggered as "well-lit condition (good lighting; not too dark, not too bright that it diminishes card details)" is a critical requirement | ![Scenario 13](assets/demo/scenario13-glare.mov) | As above |

# v0.1 OpenCV Model

![Edge Detector Model Flowchart](assets/docs/v0.1%20Python%20OpenCV%20POC.png)

1. Initialise either a webcam feed or video file
2. Apply a rectangular mask over a frame to guide the card alignment for the user (handled on the app frontend)
3. Process the frame to highlight the lines
    1. Crop and scale down the frame for faster processing
    2. Convert to grayscale
    3. Apply a **Gaussian blur** to reduce noise
    4. Apply a **Canny edge detector** to detect edges
        - The Canny edge detector is an image processing algorithm that takes in two parameters: a lower bound and upper bound
        - The algorithm will then reject pixels if the pixel gradient is below the lower bound and accept instead if the gradient is above the upper bound
        - If the gradient falls between these two bounds, the pixels will only be accepted if they are connected to pixels that are above the upper bound
        - The 5-step Canny edge detector algorithm (From [Wikipedia](https://en.wikipedia.org/wiki/Canny_edge_detector)):
            1. Apply Gaussian filterto smooth the image in order to remove the noise (Repeat of previous step)
            2. Find the intensity gradients of the image
            3. Apply gradient magnitude thresholding or lower bound cut-off suppression to get rid of spurious response to edge detection
            4. Apply double threshold to determine potential edges
            5. Track edge by [hysteresis](https://en.wikipedia.org/wiki/Hysteresis "Hysteresis"): Finalise the detection of edges by suppressing all the other edges that are weak and not connected to strong edges
        - ![Hysteresis Plot](https://scikit-image.org/docs/stable/_images/sphx_glr_plot_hysteresis_001.png)
        - The use of two thresholds with hysteresis allows more flexibility than a single-threshold approach, but general problems of thresholding approaches still apply
        - A threshold set too high can miss important information but a threshold set too low will falsely identify irrelevant information (such as noise) as important
        - It is difficult to give a generic threshold that works well on all images and no tried and tested approach to this problem yet exists as the thresholds are not dynamic during runtime
    5. Apply **Dilation** to thicken the edges so the lines are more easily found
        - Based on a set sized window, eg. a 3x3 window, if within the window, at least 1 pixel is found to be non-black, it will colour the center of the window white
        - This gives the effect of fattening the non-black regions
        - ![Dilation Effect](https://homepages.inf.ed.ac.uk/rbf/HIPR2/figs/diltbin.gif)
4. Check for lines
    1. Section off the image into 4 areas: left, right, top and bottom
    2. Use **Probabilistic Houghline Transform** to find lines in each area by checking if each potential line meets 3 conditions:
        1. The number of intersection between curves (obtained using the mathematical polar coordinate representation of a line) meets a certain threshold
            - ![Hough Lines Theory](https://docs.opencv.org/3.4/Hough_Lines_Tutorial_Theory_2.jpg)
        2. Each potential line found must be long enough
        3. The gaps between potential lines must be short enough to be considered a single line
5. Check for corners
    1. Section off the image into the 4 corners: top left, top right, bottom left and bottom right
    2. Apply **Shi-Tomasi corner detection** to find corners in the 4 corner regions from the found lines
        - At most 1 corner (set parameter) can be found by this algorithm
        - This "corner" detection algorithm does not necessarily detect corners with a certain angle
        - Rather, it finds regions of interest based on the pixel differences illustrated below, where red areas are equivalent to the "edges" and the entire green region is considered a corner if the region score _R_ is above a set parameter
            - ![Shi-Tomasi Theory](https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/shitomasi_space.png)
        - Hence, even for straight-looking lines, it may sometimes falsely identify it as a corner regardless of how strict the corner detection parameter is set (from 0 to 1)
        - The Shi-Tomasi algorithm is an improved "corner" detection algorithm to the Harris algorithm, although both face the same problems stated above
6. Card is found if at least 3 corners found and be considered a "valid frame"
7. Automatically captures the video frame after some Y milliseconds or the equivalent in some number of consecutive "valid frames" (depending on the device FPS)

## Analysis

### Pros

- Relatively fast, around 10ms to process each frame for the Python implementation (excluding frame reading due to library and device limitations)
- Dim lighting is not as bad compared to glares as the edges can still be differentiated from the background
- If colours of the card borders are similar to the background, it will be harder to detect but still possible if parameters tuned to be sensitive
- Relatively straightforward to understand and implement if utilising OpenCV, using only around 5 key image processing algorithms excluding standard image processing steps such as cropping, scaling and grayscale

### Cons

- Glare or strong light reflections causing certain areas of the frame to be completely white may hinder the detection
- If someone is holding the card in their hand or is generally in a noisy background, it can be hard to detect as well because the non-card lines can throw the model off
  - We can prompt user to place the card flat on a surface before scanning
- The "corner" detection method also detects curves, which can cause false positives for the corners even if a corner is blocked
  - This can be somewhat mitigated by adding the short delay before auto capture (some number of consecutive valid frames)

### Major Con: The OpenCV Library

The OpenCV library takes up a lot of space (PC version is 500MB-1GB), even a minimal build with the key libraries for mobile devices will likely still take up 10+MB. The existing SeaBanking app as I understand is also not currently using OpenCV. Here are some possible solutions:

#### Currently: Use OpenCV as is

For prototyping v0.1, OpenCV was used as is, with most or all of the libraries included. This allows for easy testing and tuning as there is no worry about missing features or managing requirements. However, this will not be an option in the future as the overall SDK size increment limit is currently set to be 3MB.

#### Replace OpenCV functions with custom implementation

We can replace existing functions such as Canny edge detector with a custom C++ open-sourced implementation. This will take some time and effort in researching and linking them up together to work as well as the current OpenCV prototype, which might impact performance.

#### Train a small NN model to classify a valid image (3 corners detected)

Since the rest of the steps in the Auto Capture pipeline (Card Type/Flip and Quality Check) does not use OpenCV and are instead NN-based, we can completely eliminate the use of OpenCV along the entire pipeline and just use some simple custom image input and output functions to feed into the small MNN model. This is similar to the strategy employed by the Aurora Liveness Check app, which is purely NN-based as well. This approach will also require some time and effort to gather training data and train the model.

# v0.2 Pure C++ Model (w/o OpenCV)

![v0.2 Model Development](./assets/docs/v0.2%20Pure%20C%2B%2B%20without%20OpenCV.png)

1. Image masking, cropping and resizing will be done on the app's frontend
2. After the Java preprocessing steps above, the edge detection model will receive a C++ `byteArray` or `unsigned char *` input which has already been grayscaled
3. Use the custom C++ Canny edge detector and Hough Line transform algorithm
    1. Further crop and scale down the frame for faster processing
    2. The algo applies a **Gaussian blur** to reduce noise
    3. The algo then applies a **Canny edge detector** to detect edges (full details of the Canny algorithm can be found in the v0.1 TD)
    4. I believe the algo _does not_ apply a **Dilation** to thicken the edges
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

# v0.3 Enlarged Input & Corner Scores

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

### Corner Scores

![v0.3 Corner Scores](./assets/docs/v0.3%20Corner%20Scores.png)

- A corner "accuracy" score is implemented to give more granular feedback and to allow the best frame to be picked if multiple frames with 4 corners are found
- Higher scores will be achieved if the user aligns the card as perfectly to the guided viewfinder as possible
- The model detection zone is from the red border to the yellow border; everything inside of the yellow zone is NOT scanned
- The scoring formula is calculated by checking the distance between the coordinates of the found corner against the guide view corner shown in the figure above

# v0.5 Strictly 3 Corners

![v0.5 Strictly 3 Corners](./assets/docs/v0.5%20Strictly%203%20Corners.png)

- Corners must have lines found in both side edges (instead of 1 edge)
- Mostly fixed partial and incomplete cards (< 3 corners)
- Improved stability due to slightly stricter corner logic

# v0.6 Moving Average Corners

![v0.6 Moving Average Corners](./assets/docs/v0.6%20Moving%20Average%20Corners.png)

- Store set of 4 corner coordinates in a queue (deque)
- Queue size = Sliding window size; default=3
- Average them if found (not -1), eg. if size = 3:
  - TL.x : [2, -1, 4] → [(2+4)/2] = [3] at the end
  - TL.x : [2, -1, 4, -1, -1, -1] → [2, 2, 3, 4, 4, -1]
- Fix random frame drops resetting auto capture
  - Above would have caused 4/6 rejections
- Also improves robustness to lighting changes and light backgrounds

# Final Overall Model Architecture

![Final Model Architecture](./assets/docs/v0.6%20Model%20Architecture.png)

# Final Tuneable Parameters

```cpp
struct {
    float resizedWidth = 300;                 // new width of sized down image
    float innerDetectionPercentWidth = 0.20;  // x detection inside of guideview
    float innerDetectionPercentHeight = 0.30; // y detection inside of guideview
    float sigma = 1.5;              // higher sigma for more gaussian blur
    float cannyLowerThreshold = 10; // reject if pixel gradient below threshold
    float cannyUpperThreshold = 20; // accept if pixel gradient above threshold
    int houghlineThreshold = 70;    // minimum intersections to detect a line
    float houghlineMinLineLengthRatio = 0.40; // min length of line to detect
    float houghlineMaxLineGapRatio = 0.20; // max gap between 2 potential lines
    int queueSize = 3; // moving average window of consecutive frame corners
} params;
```

| Parameter                         | Configuration [default \| min \| max] | Definition                                                       | Description                                                                                                                  | How to Decrease Strictness |
|-----------------------------------|---------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| cornerCannyLowerThreshold         | [10 \| 0 \| MAX]                      | lower pixel gradient threshold to accept                         | decreasing this will take in more details in the image (higher sensitivity, decrease strictness)                             | Decrease this              |
| cornerCannyUpperThreshold         | [20 \| 0 \| MAX]                      | upper pixel gradient threshold to accept                         | decreasing this will also accept more details in the image (higher sensitivity), but usually higher than the lower threshold | Decrease this              |
| cornerInnerDetectionPercentWidth  | [0.2 \| 0 \| 1.0]                     | amount of x area to allow detection within guideview             | amount of area to detect inside of guideview, controls how much user can zoom out and still detect the card                  | Increase this              |
| cornerInnerDetectionPercentHeight | [0.3 \| 0 \| 1.0]                     | amount of y area to allow detection within guideview             | amount of area to detect inside of guideview, controls how much user can zoom out and still detect the card                  | Increase this              |
| cornerHoughlineMaxLineGapRatio    | [0.2 \| 0 \| 1.0]                     | ratio x cornerResizedWidth                                       | increasing will cause larger gaps to be accepted and joined together to form a line                                          | Increase this              |
| cornerHoughlineMinLineLengthRatio | [0.4 \| 0 \| 1.0]                     | ratio x cornerResizedWidth, right now it’s 300 x 0.1 = 30 pixels | decreasing will cause lines that are shorter to be accepted                                                                  | Decrease this              |
| cornerHoughlineThreshold          | [70 \| 0 \| 300]                      | number of votes for a line to be accepted                        | decreasing this will lower the threshold to accept a line, making more lines and non-lines to be detected                    | Decrease this              |
| cornerQueueSize                   | [3 \| 1 \| MAX]                       | size of moving average window                                    | increasing this will allow more drop frames to still be accepted but the corners will update slower                          | Increase this              |
| cornerResizedWidth                | [300 \| 0 \| 440]                     | resize it smaller to speed up processing                         | increasing this will lose less details when downsizing the image but processing speed will be slower                         | Neutral                    |
| cornerSigma                       | [1.5 \| 0 \| 6.0]                     | the amount of blurring done to the image                         | increasing this will affect the image smoothening but take in more noise that we might think is not there                    | Neutral                    |

# Model Performance Evaluation

- Size: From 17MB with OpenCV to 42KB pure C++ version
- Speed: Around 15-20ms
- Accuracy & Stability: Did not manage to evaluate on any benchmark dataset

# Sources

## Competitor Research

- [SmartCamera Repo](https://github.com/pqpo/SmartCamera)
- [Implementation Docs](https://pqpo.me/2018/09/12/android-camera-real-time-scanning/)

### Overview of this SDK

- Library overall is written in both Java and C++ with the help of OpenCV (but the file size is big because they imported most or all of the libraries)
- The repository was last updated ~2019, using a depreciated Android camera library from ~2018-2019, around 3-4 years ago
  - Outdated camera library (depreciated in 2019 but still functions): Google’s open source CameraView
  - More updated Android Camera API: [Jetpack CameraX](https://developer.android.com/jetpack/androidx/releases/camerax)
  - CameraView & CameraX are Android APIs to use the smartphone’s cameras to get the video stream, the rest of the implementation works fine till this day
