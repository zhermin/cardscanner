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

  string app_name = "Card Scanner";

  // Load camera or video using OpenCV
  VideoCapture cap(0);
  Mat img;

  // Calculate number of frames to wait for valid frames
  int valid_frames = 0;

  cout << "Initialized " << app_name << "! Press ESC to quit" << endl;
  int i = 0;

  while (1) {
    i++;

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

    // Crop image to 480 x 301 for frame and 440 x 277 from center
    int frameW = 480, frameH = 301;
    int guideW = 440, guideH = 277;

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

    // Gaussian blur
    // Mat blurred;
    // GaussianBlur(gray, blurred, Size(0, 0), 1.5);

    // Otsu thresholding
    // Mat thresh;
    // threshold(blurred, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
    // imshow("Threshold", thresh);

    // Get the corners
    auto corners_cornerCount = cardCornerDetector.getCorners(
        cropped.data, frameW, frameH, guideW, guideH);

    vector<point_t> corners = corners_cornerCount.first;
    int cornerCount = corners_cornerCount.second;

    // Print number of corners found (score > 0)
    int cornerScore = (corners[0].score + corners[1].score + corners[2].score +
                       corners[3].score) /
                      4 * 100;
    if (cornerCount >= 3) {
      cout << "[" << cornerScore << "] " << cornerCount << " corners found!"
           << endl;
    } else {
      cout << "[" << cornerScore << "] " << cornerCount << endl;
    }

    // Draw the corners on the cropped and guideview image using OpenCV
    for (int i = 0; i < 4; i++) {
      point_t c = corners[i];
      if (c.score > 0) {
        circle(croppedBGR, Point(c.x, c.y), 5, Scalar(0, 0, 255), -1);
        circle(guideviewBGR, Point(c.x, c.y), 5, Scalar(0, 0, 255), -1);
      }
    }

    // Display the images using OpenCV
    imshow("guideview", guideviewBGR);
    imshow("cropped", croppedBGR);

    // Press ESC on keyboard to exit
    if (waitKey(1) == 27) {
      cout << "Shutting down " << app_name << endl;
      cap.release();
      break;
    };
  }

  return 0;
}
