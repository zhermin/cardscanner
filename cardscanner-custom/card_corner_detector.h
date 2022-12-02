//
// Created by Andy Jatmiko on 28/9/22.
//

#ifndef IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
#define IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
#include "houghlines/houghlines.h"
#include <queue>

class CardCornerDetector {

private:
  static line_float_t flipLine(line_float_t line);
  static float distance(point_t p1, point_t p2);

  mutable std::deque<std::vector<point_t>> cornersQueue;
  mutable std::vector<point_t> averageCorners{4, point_t{-1, -1, 0}};

public:
  struct {
    float resizedWidth = 300;        // new width of sized down image
    float detectionAreaRatio = 0.10; // ratio of detection area to image area
    float sigma = 2;                 // higher sigma for more gaussian blur
    float cannyLowerThreshold = 20;  // reject if pixel gradient below threshold
    float cannyUpperThreshold = 25;  // accept if pixel gradient above threshold
    int houghlineThreshold = 50;     // minimum intersections to detect a line
    float houghlineMinLineLengthRatio = 0.40; // min length of line to detect
    float houghlineMaxLineGapRatio = 0.20; // max gap between 2 potential lines
    int maxQueueSize = 5; // moving average of consecutive frame corners
  } params;

  std::pair<std::vector<int>, int> getCorners(unsigned char *frameByteArray,
                                              int frameWidth, int frameHeight,
                                              int guideFinderWidth,
                                              int guideFinderHeight) const;
};

#endif // IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
