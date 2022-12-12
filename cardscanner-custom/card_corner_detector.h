//
// Created by Zac Tam Zher Min.
//

#ifndef IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
#define IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
#include "houghlines/houghlines.h"
#include <queue>

class CardCornerDetector {

private:
  static unsigned char *grayscale(const unsigned char *frame, int frameWidth,
                                  int frameHeight);
  static line_float_t flipLine(line_float_t line);
  static std::pair<int, point_t>
  runningAverageEdgePoint(float x, float y, int num, point_t point);
  static float distance(point_t p1, point_t p2);
  static point_t averageCornerFromEdges(int numPoint1, point_t edgePoint1,
                                        int numPoint2, point_t edgePoint2,
                                        point_t guidePoint,
                                        float frameToGuideDist,
                                        float detectionToGuideDist);

  mutable std::vector<line_float_t> allLines;
  mutable std::deque<std::vector<point_t>> cornersQueue;
  mutable std::vector<point_t> averageCorners{4, point_t{-1, -1, 0}};

public:
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

  std::pair<std::vector<int>, int> getCorners(unsigned char *frameByteArray,
                                              int frameWidth, int frameHeight,
                                              int guideFinderWidth,
                                              int guideFinderHeight) const;

  std::vector<line_float_t> getAllLines() const { return allLines; }
};

#endif // IC_AUTO_CAPTURE_SDK_CARD_CORNER_DETECTOR_H
