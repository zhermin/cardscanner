#include "card_corner_detector.h"
#include "houghlines/houghlines.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;
CardCornerDetector cardCornerDetector;

int DEBUG = 0;

int main() {
  // $ clear && make opencv && ./main-cv.o

  string appName = "Card Scanner";

  // Load camera or video using OpenCV
  VideoCapture cap;
  cap.open(0);

  Mat img;

  cout << "Initialized " << appName << "! Press ESC to quit" << endl;
  int i = 0;

  while (1) {
    // i++; // Uncomment and change to while (i < 1) to capture only 1 frame

    // Read frame from camera or video
    cap.read(img);
    if (img.empty()) {
      cout << "Could not read frame, exiting..." << endl;
      break;
    }

    float camH = img.rows, camW = img.cols;

    // Convert to RGBA (format from Android)
    Mat rgba;
    cvtColor(img, rgba, COLOR_BGR2RGBA);

    // Crop image to 440 x 277 from center for the guideview
    int guideW = 440, guideH = 277;
    int frameW = guideW + 20, frameH = guideH + 12;

    Mat cropped, guideview;
    rgba(Rect(camW / 2 - frameW / 2, camH / 2 - frameH / 2, frameW, frameH))
        .copyTo(cropped);
    rgba(Rect(camW / 2 - guideW / 2, camH / 2 - guideH / 2, guideW, guideH))
        .copyTo(guideview);

    Mat croppedBGR, guideviewBGR;
    cvtColor(cropped, croppedBGR, COLOR_RGBA2BGR);
    cvtColor(guideview, guideviewBGR, COLOR_RGBA2BGR);

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

    // Get the corners
    auto result = cardCornerDetector.getCorners(cropped.data, frameW, frameH,
                                                guideW, guideH);

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

    // Draw the corners on the cropped and guideview image using OpenCV
    for (int i = 0; i < 4; i++) {
      int x = corners[1 + i * 2];
      int y = corners[1 + i * 2 + 1];
      if (x != -1 && y != -1) {
        circle(croppedBGR, Point(x, y), 5, Scalar(0, 0, 255), -1);
      }
    }

    // Display the images using OpenCV
    imshow("guideview", guideviewBGR);
    imshow("cropped", croppedBGR);

    // Press ESC on keyboard to exit
    if (waitKey(1) == 27) {
      cout << "Shutting down " << appName << endl;
      cap.release();
      break;
    };
  }

  return 0;
}
