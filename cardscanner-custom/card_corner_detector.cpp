//
// Created by Zac Tam Zher Min.
//

#include "card_corner_detector.h"
#include "houghlines/houghlines.h"
#include "utils.h"
#include <iostream>

line_float_t CardCornerDetector::flipLine(line_float_t line) {
  float temp = line.startx;
  line.startx = line.endx;
  line.endx = temp;
  temp = line.starty;
  line.starty = line.endy;
  line.endy = temp;
  return line;
}

std::pair<int, point_t> CardCornerDetector::getRunningAvgEdge(float x, float y,
                                                              int num,
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

point_t CardCornerDetector::getAverageCorner(int numPoint1, point_t edgePoint1,
                                             int numPoint2, point_t edgePoint2,
                                             point_t guideFinderPoint,
                                             float frameToGuideDist,
                                             float detectionToGuideDist) {

  point_t corner = {-1, -1, 0};
  if (numPoint1 == 0 || numPoint2 == 0) {
    return corner;
  }

  corner.x = (int)((edgePoint1.x + edgePoint2.x) / 2);
  corner.y = (int)((edgePoint1.y + edgePoint2.y) / 2);
  float cornerDist = distance(corner, guideFinderPoint);
  corner.score = 1 - std::min(cornerDist / frameToGuideDist,
                              cornerDist / detectionToGuideDist);
  return corner;
}

std::pair<std::vector<int>, int>
CardCornerDetector::getCorners(unsigned char *frameByteArray, int frameWidth,
                               int frameHeight, int guideFinderWidth,
                               int guideFinderHeight) const {

  auto grayscaled = Utils::grayscale(frameByteArray, frameWidth, frameHeight);

  // initialize lines vector and bounding box
  allLines.clear();
  boundingbox_t bbox = {0, 0, frameWidth, frameHeight};

  // calculate scale factor
  float scale = params.resizedWidth / (float)frameWidth;

  // use houghlines to detect lines
  HoughLineDetector(
      grayscaled, frameWidth, frameHeight, scale, scale, params.sigma,
      params.cannyLowerThreshold, params.cannyUpperThreshold, 1, PI / 180,
      params.houghlineMinLineLengthRatio * params.resizedWidth,
      params.houghlineMaxLineGapRatio * params.resizedWidth,
      params.houghlineThreshold, HOUGH_LINE_PROBABILISTIC, bbox, allLines);

  // only keep the lines in the detection regions (left, right, top, bottom)
  float detectionArea = (float)frameWidth * params.detectionAreaRatio;
  float detectionAreaRight = (float)frameWidth - detectionArea;
  float detectionAreaBottom = (float)frameHeight - detectionArea;

  std::vector<line_float_t> linesTop, linesBottom, linesLeft, linesRight;

  for (auto line : allLines) {
    if (line.startx < detectionArea && line.endx < detectionArea) {
      linesLeft.push_back(line);
    } else if (line.startx > (float)frameWidth - detectionArea &&
               line.endx > (float)frameWidth - detectionArea) {
      linesRight.push_back(line);
    } else if (line.starty < detectionArea && line.endy < detectionArea) {
      linesTop.push_back(line);
    } else if (line.starty > (float)frameHeight - detectionArea &&
               line.endy > (float)frameHeight - detectionArea) {
      linesBottom.push_back(line);
    }
  }

  // check in all 4 corners for points that are within the detection area
  // and save the running average of the x and y coordinates
  // also orientate lines so they go from left to right and top to bottom
  point_t pointTopStart, pointTopEnd, pointBottomStart, pointBottomEnd,
      pointLeftStart, pointLeftEnd, pointRightStart, pointRightEnd;
  int numTopStart = 0, numTopEnd = 0, numBottomStart = 0, numBottomEnd = 0,
      numLeftStart = 0, numLeftEnd = 0, numRightStart = 0, numRightEnd = 0;

  for (auto line : linesTop) {
    if (line.startx > line.endx) {
      line = flipLine(line);
    }
    if (line.startx <= detectionArea && line.starty <= detectionArea) {
      std::tie(numTopStart, pointTopStart) = getRunningAvgEdge(
          line.startx, line.starty, numTopStart, pointTopStart);
    }
    if (line.endx >= detectionAreaRight && line.endy <= detectionArea) {
      std::tie(numTopEnd, pointTopEnd) =
          getRunningAvgEdge(line.endx, line.endy, numTopEnd, pointTopEnd);
    }
  }

  for (auto line : linesBottom) {
    if (line.startx > line.endx) {
      line = flipLine(line);
    }
    if (line.startx <= detectionArea && line.starty >= detectionAreaBottom) {
      std::tie(numBottomStart, pointBottomStart) = getRunningAvgEdge(
          line.startx, line.starty, numBottomStart, pointBottomStart);
    }
    if (line.endx >= detectionAreaRight && line.endy >= detectionAreaBottom) {
      std::tie(numBottomEnd, pointBottomEnd) =
          getRunningAvgEdge(line.endx, line.endy, numBottomEnd, pointBottomEnd);
    }
  }

  for (auto line : linesLeft) {
    if (line.starty > line.endy) {
      line = flipLine(line);
    }
    if (line.startx <= detectionArea && line.starty <= detectionArea) {
      std::tie(numLeftStart, pointLeftStart) = getRunningAvgEdge(
          line.startx, line.starty, numLeftStart, pointLeftStart);
    }
    if (line.endx <= detectionArea && line.endy >= detectionAreaBottom) {
      std::tie(numLeftEnd, pointLeftEnd) =
          getRunningAvgEdge(line.endx, line.endy, numLeftEnd, pointLeftEnd);
    }
  }

  for (auto line : linesRight) {
    if (line.starty > line.endy) {
      line = flipLine(line);
    }
    if (line.startx >= detectionAreaRight && line.starty <= detectionArea) {
      std::tie(numRightStart, pointRightStart) = getRunningAvgEdge(
          line.startx, line.starty, numRightStart, pointRightStart);
    }
    if (line.endx >= detectionAreaRight && line.endy >= detectionAreaBottom) {
      std::tie(numRightEnd, pointRightEnd) =
          getRunningAvgEdge(line.endx, line.endy, numRightEnd, pointRightEnd);
    }
  }

  // store the corners of the frame and detection zones
  int guideFinderGapX = (frameWidth - guideFinderWidth) / 2;
  int guideFinderGapY = (frameHeight - guideFinderHeight) / 2;

  point_t guideFinderTopLeft = {guideFinderGapX, guideFinderGapY};
  point_t guideFinderTopRight = {frameWidth - guideFinderGapX, guideFinderGapY};
  point_t guideFinderBottomRight = {frameWidth - guideFinderGapX,
                                    frameHeight - guideFinderGapY};
  point_t guideFinderBottomLeft = {guideFinderGapX,
                                   frameHeight - guideFinderGapY};

  point_t detectionTopLeft = {(int)detectionArea, (int)detectionArea};

  float frameToGuideDist = distance(point_t{0, 0}, guideFinderTopLeft);
  float detectionToGuideDist = distance(detectionTopLeft, guideFinderTopLeft);

  // the corner points are the average of the 2 edge points if they exist
  point_t cornerTopLeft = getAverageCorner(
      numTopStart, pointTopStart, numLeftStart, pointLeftStart,
      guideFinderTopLeft, frameToGuideDist, detectionToGuideDist);

  point_t cornerTopRight = getAverageCorner(
      numTopEnd, pointTopEnd, numRightStart, pointRightStart,
      guideFinderTopRight, frameToGuideDist, detectionToGuideDist);

  point_t cornerBottomRight = getAverageCorner(
      numBottomEnd, pointBottomEnd, numRightEnd, pointRightEnd,
      guideFinderBottomRight, frameToGuideDist, detectionToGuideDist);

  point_t cornerBottomLeft = getAverageCorner(
      numBottomStart, pointBottomStart, numLeftEnd, pointLeftEnd,
      guideFinderBottomLeft, frameToGuideDist, detectionToGuideDist);

  // add corners to queue in order of TL, TR, BR, BL
  cornersQueue.push_back(
      {cornerTopLeft, cornerTopRight, cornerBottomRight, cornerBottomLeft});

  // remove the oldest set of corners from the queue
  if (cornersQueue.size() > params.queueSize) {
    cornersQueue.pop_front();
  }

  // get the average of the corners
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

  // append the corner count and corner coordinates to an output int vector
  std::vector<int> foundCorners;
  float cornerScore = 0;

  foundCorners.push_back(cornerCount);
  for (int i = 0; i < 4; i++) {
    foundCorners.push_back(averageCorners[i].x);
    foundCorners.push_back(averageCorners[i].y);
    cornerScore += averageCorners[i].score;
  }

  // return the vector of found corners
  delete grayscaled;
  return make_pair(foundCorners, (int)(cornerScore / 4 * 100));
}