# CardScanner Custom C++ Version

Custom C++ Version to perform edge/corner detection on cards to replace the OpenCV library due to extremely limited memory allotment on mobile devices.

# Introduction

This TD showcases the next version v0.2 of the card edge/corner detector. The main improvement over version v0.1 is the removal of all OpenCV dependencies and instead using an open-sourced custom implementation of the line detection algorithm in C++.

The primary goal of removing the dependency on OpenCV is to reduce the eventual app size. As elaborated in the previous TD for v0.1, the OpenCV library takes up a huge amount of storage. Even the mobile bundle requires upwards of 10MB, which severely exceeds the current set limit of 3MB for the entire SDK, including the other models in the auto-scanning pipeline.

However, the performance of the model might worsen in this version because some OpenCV functions previously used were harder to find custom implementations for, such as the Shi-Tomasi corner detection algorithm. Instead, tentatively, a simpler corner detection method is used in this version.

## Model Development (Pure C++ Implementation)

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

## Model Performance Evaluation

No extensive testing was done on the model performance yet because the focus was on removing the OpenCV dependency. The model's performance will likely suffer because the OpenCV version might have been better written and more robust. However, this is naturally the cost of removing the OpenCV library.

### Metrics

- Size: Total size of the model and if including imported libraries, if any
- Speed: Time to process each frame
- Accuracy: Auto-capturing when it is supposed to ("Predicted" Positives)
- Stability: NOT auto-capturing when it is not supposed to ("Predicted" Negatives)

|      |                 |                     |       |          |           |
| ---- | --------------- | ------------------- | ----- | -------- | --------- |
|      | Pure Model Size | SDK Size with Model | Speed | Accuracy | Stability |
| v0.1 |                 |                     |       |          |           |
| v0.2 |                 |                     |       |          |           |
| v0.3 |                 |                     |       |          |           |
| v0.4 |                 |                     |       |          |           |
