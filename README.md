# SmartCamera by PQPO

[SmartCamera Repo](https://github.com/pqpo/SmartCamera)  
[Implementation Docs](https://pqpo.me/2018/09/12/android-camera-real-time-scanning/)

## Overview of this SDK
- Last updated ~2019, using tech from ~2018-2019, around 3-4 years ago
    - Outdated library (depreciated in 2019): Google’s open source CameraView
    - More updated Android Camera API: [Jetpack CameraX](https://developer.android.com/jetpack/androidx/releases/camerax)
- CameraView & CameraX are Android APIs to use the smartphone’s cameras to get the video stream
- A bit hard to fully understand the code because it is mostly in Java
- If want to implement for iOS or cross-platform will need to find the SDKs/APIs for the camera stream for iOS and Android for the correct platform (eg. Swift for iOS or React Native/Flutter for cross-platform)

### Pros
- Simple and fast, using only OpenCV to process image without deep learning, good for mobile
- Can use the same techniques to detect edges using OpenCV for our own implementation

### Cons
- Project is not updated anymore and is using old libraries
- Only works for Android because it is using Android-specific APIs

## How It Works (Based on my own understanding of PQPO’s docs)
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
