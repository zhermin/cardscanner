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
    float resizedWidth = 300;        // new width of sized down image
    float detectionAreaRatio = 0.08; // ratio of detection area to image area
    float sigma = 3;                 // higher sigma for more gaussian blur
    int cannyLowerThreshold = 20;    // reject if pixel gradient below threshold
    int cannyUpperThreshold = 25;    // accept if pixel gradient above threshold
    int houghlineThreshold = 50;     // minimum intersections to detect a line
    float houghlineMinLineLengthRatio = 0.30; // min length of line to detect
    float houghlineMaxLineGapRatio = 0.30; // max gap between 2 potential lines
  } params;

  pair<vector<point_t>, int> getCorners(unsigned char *frameByteArray,
                                        int frameWidth, int frameHeight,
                                        int guideFinderWidth,
                                        int guideFinderHeight);
};

#endif // IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
