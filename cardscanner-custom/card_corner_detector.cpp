//
// Created by Zac Tam Zher Min.
//

#include "card_corner_detector.h"
#include "houghlines/houghlines.h"
#include <iostream>

std::pair<std::vector<int>, int>
CardCornerDetector::getCorners(unsigned char *frameByteArray, int frameWidth,
                               int frameHeight, int guideFinderWidth,
                               int guideFinderHeight) const {

  auto grayscaled = grayscale(frameByteArray, frameWidth, frameHeight);

  // clear the lines found in the previous frame and initialize bounding box
  allLines.clear();
  boundingbox_t bbox = {0, 0, frameWidth, frameHeight};

  // calculate downscaling factor for faster houghline detection
  float scale = params.resizedWidth / (float)frameWidth;

  // this open-source library performs gaussian -> canny -> houghline detection
  // gaussian for smoothing, canny for edge map, for houghline to find lines
  // output is a vector of custom line_float_t{startx, starty, endx, endy}
  // https://github.com/frotms/line_detector/tree/master/houghlines
  HoughLineDetector(
      grayscaled, frameWidth, frameHeight, scale, scale, params.sigma,
      params.cannyLowerThreshold, params.cannyUpperThreshold, 1, PI / 180,
      params.houghlineMinLineLengthRatio * params.resizedWidth,
      params.houghlineMaxLineGapRatio * params.resizedWidth,
      params.houghlineThreshold, HOUGH_LINE_PROBABILISTIC, bbox, allLines);

  // only keep lines with start/end line coordinates in the 4 side regions
  int cornerMarginX = (frameWidth - guideFinderWidth) / 2;
  int cornerMarginY = (frameHeight - guideFinderHeight) / 2;

  float detectionAreaLeft =
      (float)cornerMarginX +
      (float)guideFinderWidth / 2 * params.innerDetectionPercentWidth;
  float detectionAreaTop =
      (float)cornerMarginY +
      (float)guideFinderHeight / 2 * params.innerDetectionPercentHeight;
  float detectionAreaRight = (float)frameWidth - detectionAreaLeft;
  float detectionAreaBottom = (float)frameHeight - detectionAreaTop;

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

  // first orientate each line to be from left to right and top to bottom
  // each of the 4 corner regions will have 2 points contributed by 2 edges
  // and each of the total 8 points will be the running average coordinates
  point_t pointTopStart, pointTopEnd, pointBottomStart, pointBottomEnd,
      pointLeftStart, pointLeftEnd, pointRightStart, pointRightEnd;
  int numTopStart = 0, numTopEnd = 0, numBottomStart = 0, numBottomEnd = 0,
      numLeftStart = 0, numLeftEnd = 0, numRightStart = 0, numRightEnd = 0;

  for (auto line : linesTop) {
    if (line.startx > line.endx) {
      line = flipLine(line);
    }
    if (line.startx <= detectionAreaLeft && line.starty <= detectionAreaTop) {
      std::tie(numTopStart, pointTopStart) = runningAverageEdgePoint(
          line.startx, line.starty, numTopStart, pointTopStart);
    }
    if (line.endx >= detectionAreaRight && line.endy <= detectionAreaTop) {
      std::tie(numTopEnd, pointTopEnd) =
          runningAverageEdgePoint(line.endx, line.endy, numTopEnd, pointTopEnd);
    }
  }

  for (auto line : linesBottom) {
    if (line.startx > line.endx) {
      line = flipLine(line);
    }
    if (line.startx <= detectionAreaLeft &&
        line.starty >= detectionAreaBottom) {
      std::tie(numBottomStart, pointBottomStart) = runningAverageEdgePoint(
          line.startx, line.starty, numBottomStart, pointBottomStart);
    }
    if (line.endx >= detectionAreaRight && line.endy >= detectionAreaBottom) {
      std::tie(numBottomEnd, pointBottomEnd) = runningAverageEdgePoint(
          line.endx, line.endy, numBottomEnd, pointBottomEnd);
    }
  }

  for (auto line : linesLeft) {
    if (line.starty > line.endy) {
      line = flipLine(line);
    }
    if (line.startx <= detectionAreaLeft && line.starty <= detectionAreaTop) {
      std::tie(numLeftStart, pointLeftStart) = runningAverageEdgePoint(
          line.startx, line.starty, numLeftStart, pointLeftStart);
    }
    if (line.endx <= detectionAreaLeft && line.endy >= detectionAreaBottom) {
      std::tie(numLeftEnd, pointLeftEnd) = runningAverageEdgePoint(
          line.endx, line.endy, numLeftEnd, pointLeftEnd);
    }
  }

  for (auto line : linesRight) {
    if (line.starty > line.endy) {
      line = flipLine(line);
    }
    if (line.startx >= detectionAreaRight && line.starty <= detectionAreaTop) {
      std::tie(numRightStart, pointRightStart) = runningAverageEdgePoint(
          line.startx, line.starty, numRightStart, pointRightStart);
    }
    if (line.endx >= detectionAreaRight && line.endy >= detectionAreaBottom) {
      std::tie(numRightEnd, pointRightEnd) = runningAverageEdgePoint(
          line.endx, line.endy, numRightEnd, pointRightEnd);
    }
  }

  // store corners of the frame and detection zones to calculate corner score
  int guideFinderGapX = (frameWidth - guideFinderWidth) / 2;
  int guideFinderGapY = (frameHeight - guideFinderHeight) / 2;

  point_t guideFinderTopLeft = {guideFinderGapX, guideFinderGapY};
  point_t guideFinderTopRight = {frameWidth - guideFinderGapX, guideFinderGapY};
  point_t guideFinderBottomRight = {frameWidth - guideFinderGapX,
                                    frameHeight - guideFinderGapY};
  point_t guideFinderBottomLeft = {guideFinderGapX,
                                   frameHeight - guideFinderGapY};

  point_t detectionTopLeft = {(int)detectionAreaLeft, (int)detectionAreaTop};

  float frameToGuideDist = distance(point_t{0, 0}, guideFinderTopLeft);
  float detectionToGuideDist = distance(detectionTopLeft, guideFinderTopLeft);

  // only if both edge points found, we average the 2 points to get the corner
  point_t cornerTopLeft = averageCornerFromEdges(
      numTopStart, pointTopStart, numLeftStart, pointLeftStart,
      guideFinderTopLeft, frameToGuideDist, detectionToGuideDist);

  point_t cornerTopRight = averageCornerFromEdges(
      numTopEnd, pointTopEnd, numRightStart, pointRightStart,
      guideFinderTopRight, frameToGuideDist, detectionToGuideDist);

  point_t cornerBottomRight = averageCornerFromEdges(
      numBottomEnd, pointBottomEnd, numRightEnd, pointRightEnd,
      guideFinderBottomRight, frameToGuideDist, detectionToGuideDist);

  point_t cornerBottomLeft = averageCornerFromEdges(
      numBottomStart, pointBottomStart, numLeftEnd, pointLeftEnd,
      guideFinderBottomLeft, frameToGuideDist, detectionToGuideDist);

  // this set of 4 corners in the queue is considered 1 frame
  cornersQueue.push_back(
      {cornerTopLeft, cornerTopRight, cornerBottomRight, cornerBottomLeft});

  if (cornersQueue.size() > params.queueSize) {
    cornersQueue.pop_front();
  }

  // sliding window average of the 4 corners from the last queueSize frames
  int cornerCount = 0;

  for (int i = 0; i < 4; i++) {
    float num = 0, sumX = 0, sumY = 0, sumScore = 0;

    for (auto queueCorner = cornersQueue.crbegin();
         queueCorner != cornersQueue.crend(); queueCorner++) {
      if (queueCorner->at(i).x != -1 && queueCorner->at(i).y != -1) {
        sumX += (float)queueCorner->at(i).x;
        sumY += (float)queueCorner->at(i).y;
        sumScore += queueCorner->at(i).score;
        num++;
      }
    }

    if (num > 0) {
      averageCorners[i] = {(int)(sumX / num), (int)(sumY / num),
                           sumScore / num};
      cornerCount++;
    } else {
      averageCorners[i] = {-1, -1, 0};
    }
  }

  // the format of this int vector is synced up with the mobile FE side
  std::vector<int> foundCorners;
  float cornerScore = 0;

  foundCorners.push_back(cornerCount);
  for (int i = 0; i < 4; i++) {
    foundCorners.push_back(averageCorners[i].x);
    foundCorners.push_back(averageCorners[i].y);
    cornerScore += averageCorners[i].score;
  }

  delete grayscaled;
  return make_pair(foundCorners, (int)(cornerScore / 4 * 100));
}

unsigned char *CardCornerDetector::grayscale(const unsigned char *frame,
                                             int frameWidth, int frameHeight) {
  unsigned char *grayscaled = new unsigned char[frameWidth * frameHeight];
  for (int i = 0; i < frameWidth * frameHeight; i++) {
    grayscaled[i] =
        (unsigned char)(0.299 * frame[i * 4] + 0.587 * frame[i * 4 + 1] +
                        0.114 * frame[i * 4 + 2]);
  }
  return grayscaled;
}

line_float_t CardCornerDetector::flipLine(line_float_t line) {
  float temp = line.startx;
  line.startx = line.endx;
  line.endx = temp;
  temp = line.starty;
  line.starty = line.endy;
  line.endy = temp;
  return line;
}

std::pair<int, point_t>
CardCornerDetector::runningAverageEdgePoint(float x, float y, int num,
                                            point_t point) {

  if (num == 0) {
    point = {(int)x, (int)y, 0};
  } else {
    point.x = (point.x * num + (int)x) / (num + 1);
    point.y = (point.y * num + (int)y) / (num + 1);
  }

  return std::make_pair(num + 1, point);
}

float CardCornerDetector::distance(point_t p1, point_t p2) {
  // no need to root the distance as we are getting a ratio
  return (float)(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

point_t CardCornerDetector::averageCornerFromEdges(
    int numPoint1, point_t edgePoint1, int numPoint2, point_t edgePoint2,
    point_t guidePoint, float frameToGuideDist, float detectionToGuideDist) {

  point_t corner = {-1, -1, 0};
  if (numPoint1 == 0 || numPoint2 == 0) {
    return corner;
  }

  corner.x = (int)((edgePoint1.x + edgePoint2.x) / 2);
  corner.y = (int)((edgePoint1.y + edgePoint2.y) / 2);
  float cornerDist = distance(corner, guidePoint);
  corner.score = 1 - std::min(cornerDist / frameToGuideDist,
                              cornerDist / detectionToGuideDist);
  return corner;
}