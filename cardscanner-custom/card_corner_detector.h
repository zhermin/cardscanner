//
// Created by Andy Jatmiko on 28/9/22.
//

#ifndef IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
#define IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
#include "houghlines/houghlines.h"

using namespace std;

class CardCornerDetector {

private:
  line_float_t flipLine(line_float_t line);
  float distance(point_t p1, point_t p2);

public:
  struct {
    float resizedWidth = 300; // new width of sized down image for faster processing
    float detectionAreaRatio = 0.10; // ratio of the detection area to the image area
    float sigma = 1.0;              // higher sigma for more gaussian blur
    int cannyLowerThreshold = 10; // rejected if pixel gradient below lower threshold
    int cannyUpperThreshold = 30; // accepted if pixel gradient above upper threshold
    int houghlineThreshold = 60;  // minimum intersections to detect a line
    float houghlineMinLineLengthRatio = 0.10; // minimum length of a line to detect
    float houghlineMaxLineGapRatio = 0.10; // maximum gap between two potential lines
  } params;

  pair<vector<point_t>, int> getCorners(unsigned char *frameByteArray,
                                    int frameWidth, int frameHeight,
                                    int guideFinderWidth,
                                    int guideFinderHeight);
};

#endif // IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
