// $ clear && make opencv && ./main-cv.o

#include "card_corner_detector.h"
#include "houghlines/houghlines.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
CardCornerDetector cardCornerDetector;

int main() {
  string appName = "Card Scanner";

  // Load camera or video using OpenCV
  VideoCapture cap(1);

  Mat img;

  cout << "Initialized " << appName << "! Press ESC to quit" << endl;
  int i = 0;

  while (1) {
    // i++; // Uncomment and change to while (i < 1) to capture only 1 frame

    // Values synced up from Mobile Frontend side
    int guideFinderWidth = 440, guideFinderHeight = 277;
    float extendedPercentageWidth = 0.05, extendedPercentageHeight = 0.05;

    int frameWidth = guideFinderWidth * (1 + extendedPercentageWidth * 2),
        frameHeight = guideFinderHeight * (1 + extendedPercentageHeight * 2);

    // Read frame from camera or video
    cap.read(img);
    if (img.empty()) {
      cout << "Could not read frame, exiting..." << endl;
      break;
    }

    float camHeight = img.rows, camWidth = img.cols;

    // Convert to RGBA (format from Android)
    Mat rgba;
    cvtColor(img, rgba, COLOR_BGR2RGBA);

    Mat cropped, guideview;
    rgba(Rect(camWidth / 2 - frameWidth / 2, camHeight / 2 - frameHeight / 2,
              frameWidth, frameHeight))
        .copyTo(cropped);
    rgba(Rect(camWidth / 2 - guideFinderWidth / 2,
              camHeight / 2 - guideFinderHeight / 2, guideFinderWidth,
              guideFinderHeight))
        .copyTo(guideview);

    Mat croppedBGR, guideviewBGR;
    cvtColor(cropped, croppedBGR, COLOR_RGBA2BGR);
    cvtColor(guideview, guideviewBGR, COLOR_RGBA2BGR);

    /* OpenCV Experiments: RGB Channels, Average Grayscale Intensity, Histogram,
    Gaussian Blur, Otsu Thresholding, etc. */

    // Display RGB channels separately
    // Mat gray;
    // cvtColor(croppedBGR, gray, COLOR_BGR2GRAY);
    // Mat bgr[3];
    // split(croppedBGR, bgr);
    // imshow("Blue", bgr[0]);
    // imshow("Green", bgr[1]);
    // imshow("Red", bgr[2]);
    // imshow("Gray", gray);

    // Calculate average grayscale intensity
    // Scalar intensity = mean(gray);
    // float avg = intensity.val[0];
    // cout << "Average grayscale intensity: " << avg << endl;

    // // Plot the grayscale histogram
    // int histSize = 256;
    // float range[] = {0, 256};
    // const float *histRange = {range};
    // bool uniform = true, accumulate = false;
    // Mat hist;
    // calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform,
    //          accumulate);
    // int hist_w = 512, hist_h = 400;
    // int bin_w = cvRound((double)hist_w / histSize);
    // Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
    // normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    // for (int i = 1; i < histSize; i++) {
    //   line(histImage,
    //        Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
    //        Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
    //        Scalar(255, 0, 0), 2, 8, 0);
    // }
    // imshow("Histogram", histImage);

    // Gaussian blur
    // Mat blurred;
    // GaussianBlur(gray, blurred, Size(0, 0), 1.5);

    // Otsu thresholding
    // Mat thresh;
    // threshold(blurred, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
    // imshow("Threshold", thresh);

    // Get the corners in a tuple (corners, score)
    auto result =
        cardCornerDetector.getCorners(cropped.data, frameWidth, frameHeight,
                                      guideFinderWidth, guideFinderHeight);

    auto corners = result.first;
    int cornerCount = corners[0];
    auto cornerScore = result.second;

    // Print number of corners found (score > 0)
    if (cornerCount >= 3) {
      cout << "[" << cornerScore << "] " << cornerCount << " corners found!"
           << endl;
    } else {
      cout << "[" << cornerScore << "] " << cornerCount << endl;
    }

    // Calculate the guideview and detection corners
    int cornerMarginX = (frameWidth - guideFinderWidth) / 2;
    int cornerMarginY = (frameHeight - guideFinderHeight) / 2;

    float detectionAreaLeft =
        (float)cornerMarginX +
        (float)guideFinderWidth / 2 *
            cardCornerDetector.params.innerDetectionPercentWidth;
    float detectionAreaTop =
        (float)cornerMarginY +
        (float)guideFinderHeight / 2 *
            cardCornerDetector.params.innerDetectionPercentHeight;
    float detectionAreaRight = (float)frameWidth - detectionAreaLeft;
    float detectionAreaBottom = (float)frameHeight - detectionAreaTop;

    point_t guideFinderTopLeft = {cornerMarginX, cornerMarginY};
    point_t guideFinderTopRight = {frameWidth - cornerMarginX, cornerMarginY};
    point_t guideFinderBottomLeft = {cornerMarginX,
                                     frameHeight - cornerMarginY};
    point_t guideFinderBottomRight = {frameWidth - cornerMarginX,
                                      frameHeight - cornerMarginY};

    point_t detectionTopLeft = {(int)detectionAreaLeft, (int)detectionAreaTop};
    point_t detectionTopRight = {(int)detectionAreaRight,
                                 (int)detectionAreaTop};
    point_t detectionBottomLeft = {(int)detectionAreaLeft,
                                   (int)detectionAreaBottom};
    point_t detectionBottomRight = {(int)detectionAreaRight,
                                    (int)detectionAreaBottom};

    // Retrieve all the found lines
    vector<line_float_t> allLines = cardCornerDetector.getAllLines();
    std::vector<line_float_t> linesTop, linesBottom, linesLeft, linesRight;

    for (auto line : allLines) {
      if (line.startx <= detectionAreaLeft && line.endx <= detectionAreaLeft) {
        linesLeft.push_back(line);
      } else if (line.startx >= detectionAreaRight &&
                 line.endx >= detectionAreaRight) {
        linesRight.push_back(line);
      } else if (line.starty <= detectionAreaTop &&
                 line.endy <= detectionAreaTop) {
        linesTop.push_back(line);
      } else if (line.starty >= detectionAreaBottom &&
                 line.endy >= detectionAreaBottom) {
        linesBottom.push_back(line);
      }
    }

    // Combine the filtered region lines
    vector<line_float_t> filteredLines;
    filteredLines.insert(filteredLines.end(), linesTop.begin(), linesTop.end());
    filteredLines.insert(filteredLines.end(), linesBottom.begin(),
                         linesBottom.end());
    filteredLines.insert(filteredLines.end(), linesLeft.begin(),
                         linesLeft.end());
    filteredLines.insert(filteredLines.end(), linesRight.begin(),
                         linesRight.end());

    //--- BEGIN DRAWING ---//

    // Draw on a new image either all the lines or the filtered lines
    vector<line_float_t> linesToDraw = allLines;

    Mat linesImg = Mat::zeros(croppedBGR.size(), CV_8UC3);
    for (int i = 0; i < linesToDraw.size(); i++) {
      line(linesImg, Point(linesToDraw[i].startx, linesToDraw[i].starty),
           Point(linesToDraw[i].endx, linesToDraw[i].endy),
           Scalar(255, 255, 255), 1, LINE_AA);
    }

    // Draw the guideview corners
    circle(linesImg, Point(guideFinderTopLeft.x, guideFinderTopLeft.y), 1,
           Scalar(0, 255, 255), -1);
    circle(linesImg, Point(guideFinderTopRight.x, guideFinderTopRight.y), 1,
           Scalar(0, 255, 255), -1);
    circle(linesImg, Point(guideFinderBottomLeft.x, guideFinderBottomLeft.y), 1,
           Scalar(0, 255, 255), -1);
    circle(linesImg, Point(guideFinderBottomRight.x, guideFinderBottomRight.y),
           1, Scalar(0, 255, 255), -1);

    // Draw the detection corners
    circle(linesImg, Point(detectionTopLeft.x, detectionTopLeft.y), 1,
           Scalar(0, 255, 0), -1);
    circle(linesImg, Point(detectionTopRight.x, detectionTopRight.y), 1,
           Scalar(0, 255, 0), -1);
    circle(linesImg, Point(detectionBottomLeft.x, detectionBottomLeft.y), 1,
           Scalar(0, 255, 0), -1);
    circle(linesImg, Point(detectionBottomRight.x, detectionBottomRight.y), 1,
           Scalar(0, 255, 0), -1);

    // Draw the corners on the cropped and guideview image
    for (int i = 0; i < 4; i++) {
      int x = corners[1 + i * 2];
      int y = corners[1 + i * 2 + 1];
      if (x != -1 && y != -1) {
        circle(croppedBGR, Point(x, y), 5, Scalar(0, 0, 255), -1);
        circle(linesImg, Point(x, y), 3, Scalar(0, 0, 255), -1);
      }
    }

    //--- END DRAWING ---//

    // Display the images
    imshow("guideview", guideviewBGR);
    imshow("cropped", croppedBGR);
    imshow("lines", linesImg);

    // Press ESC on the keyboard to exit
    if (waitKey(1) == 27) {
      cout << "Shutting down " << appName << endl;
      cap.release();
      break;
    };
  }

  return 0;
}
