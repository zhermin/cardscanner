# CardScanner
A simple proof-of-concept real-time card scanner using Python and OpenCV based on an Android implementation by PQPO. 

## Application Flow
1. Initialize either a video file or webcam feed
2. Apply a rectangular mask over a frame to guide the card alignment for the user
    - [PARAM] Percentage of frame to be obscured
3. Process the frame to highlight the lines
    1. Crop and scale down the frame for faster processing
        - [PARAM] Size of scaled down frame
    2. Convert to grayscale
    3. Apply a Gaussian blur to reduce noise
        - [PARAM] Larger kernel size will blur image more
    4. Apply a Canny edge detector to detect edges
        - [PARAM] Rejects pixel if pixel gradient below lower threshold
        - [PARAM] Accepts pixel if pixel gradient above upper threshold
        - If between threshold, pixel accepted only if connected to pixel above upper threshold
    5. Apply Dilation to strengthen the edges
        - [PARAM] Larger kernel size dilates the lines more
4.  Check for lines
    1. Section off the image into 4 areas: left, right, top and bottom
        - [PARAM] Percentage of inner area to perform detection
    2. Use Probabilistic Hough Transform to find lines in each area
        - [PARAM] Each potential line must be long enough
        - [PARAM] Gaps between potential lines must be short enough
5.  Check for corners
    1. Section off the image into the 4 corners: top left, top right, bottom left and bottom right
    2. Apply Shi-Tomasi corner detection to find corners in the found lines
        - [PARAM] Higher ratio means stricter detection
    3. There will be only max 1 corner found for each corner
6.  Conditions for a card: at least 3 corners found (based on PRD requirements)
    1.  Give visual feedback for each line or corner found separately
7.  Automatically captures the video frame after some Y milliseconds or the equivalent in some number of frames

## Analysis

### Pros
- Relatively fast, around 10ms to process each frame (excluding frame reading due to library and device limitations)
- Simple to implement and understand, will help when writing the custom functions to do edge detection on mobile devices
- Uses the OpenCV library, which has support for both Python and C++, allowing a smoother transition to the custom Android APK POC
- Dim lighting is not as bad compared to glares as the edges can still be differentiated from the background
- If colours of the card borders are similar to the background, it will be harder to detect but still possible if parameters tuned to be sensitive

### Cons
- Glare or strong light reflections causing the video feed to be completely white will hinder the detection
- If someone is holding the card in their hand or is generally in a noisy background, it can be hard to detect as well
    - We can prompt user to place the card flat on a surface before scanning
- The "corner" detection method also detects curves, which can cause false positives for the corners even if a corner is blocked
    - This can be somewhat mitigated by adding the short delay before auto capture (some number of consecutive valid frames)

## Setup

### Installation
```bash
git clone gitlab@git.garena.com:shopee-ds/kyc/icas/intern/zac-tam-zher-min/cardscanner.git
cd cardscanner
pip install -r requirements.txt
```

### Usage
```bash
python main.py
    --cam <cam ID, default=0>
    --preview <0/1 off/on mini-preview of the detected edges>
    --file <path to image or video file instead of webcam>
```

### Configs
Changes to parameters can be done in the `main.py` script directly. The configs can be pulled out into a YAML config file but for simplicity, this was done instead. 

**Note:** Ratio params are based off of the max_size param
```python
PARAMS = {
    "max_size": 300,  # scaled down image for faster processing
    "mask_aspect_ratio": (86, 54),  # CR80 standard card size is 86mm x 54mm
    "frame_scaling_factor": 0.6,  # ratio of unmasked area to the entire frame
    "gaussian_blur_radius": 5,  # higher radius = more blur
    "canny_lowerthreshold_ratio": 0.03,  # rejected if pixel gradient below lower threshold
    "canny_upperthreshold_ratio": 0.10,  # accepted if pixel gradient above upper threshold
    "dilate_structing_element_size": 3,  # larger kernel = thicker lines
    "houghlines_threshold_ratio": 0.3,  # minimum intersections to detect a line
    "houghlines_min_line_length": 0.3,  # minimum length of a line
    "houghlines_max_line_gap": 0.01,  # maximum gap between two points to form a line
    "area_detection_ratio": 0.15,  # ratio of the detection area to the image area
    "corner_quality_ratio": 0.9,  # higher value = stricter corner detection
    "y_milliseconds": 200,  # number of milliseconds to wait for valid frames
}
```

## Based On: SmartCamera by PQPO

[SmartCamera Repo](https://github.com/pqpo/SmartCamera)  
[Implementation Docs](https://pqpo.me/2018/09/12/android-camera-real-time-scanning/)

### Overview of this SDK
- Last updated ~2019, using tech from ~2018-2019, around 3-4 years ago
    - Outdated library (depreciated in 2019): Google’s open source CameraView
    - More updated Android Camera API: [Jetpack CameraX](https://developer.android.com/jetpack/androidx/releases/camerax)
- CameraView & CameraX are Android APIs to use the smartphone’s cameras to get the video stream
- A bit hard to fully understand the code because it is mostly in Java
- If want to implement for iOS or cross-platform will need to find the SDKs/APIs for the camera stream for iOS and Android for the correct platform (eg. Swift for iOS or React Native/Flutter for cross-platform)

#### Pros
- Simple and fast, using only OpenCV to process image without deep learning, good for mobile
- Can use the same techniques to detect edges using OpenCV for our own implementation

#### Cons
- Project is not updated anymore and is using old libraries
- Only works for Android because it is using Android-specific APIs

### How It Works (Based on my own understanding of PQPO’s docs)
1. Capture the video stream from smartphone using the Android Camera API
2. The output from the API is a frame buffer
    1. Each frame is in the format “YCbCr_420_SP (NV21)” or “YUV 4:2:0 sampling”
    2. To allow OpenCV to process the YUV formatted image, convert it to “Mat format”
    3. Convert it to a greyscale or black & white image and rotate the image if needed
3. Crop out the rectangular area from the image based on (maskX, maskY, maskWidth, maskHeight) and scale it down using (scaleRatio)
4. Perform Edge Detection & Extraction algorithm using OpenCV
    1. Apply Gaussian Blur `GaussianBlur()` to remove noise
    2. Use the `Canny()` operation to detect edges
    3. Use the `dilate()` operation to strengthen edges
    4. Binarise and threshold the image using `threshold()` to remove interference
5. Verify if image is a card
    1. Divide image into 4 detection areas (top, bottom, left, right), say, 10% away from the edges of the rectangular mask
    2. Detect all the lines in the 4 general areas
    3. Determine whether there are enough lines that meet the conditions for each area by checking their lengths and angles (vertical/horizontal)
6. Finally if all 4 areas pass the checks, the image is considered a card and a photo is automatically captured and saved
