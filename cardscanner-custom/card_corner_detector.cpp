#include "card_corner_detector.h"
#include "houghlines/houghlines.h"
#include "utils.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// #include <android/log.h>
// #define TAG "AUTO_CAPTURE_PROCESSOR"
// #define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using namespace std;

line_float_t CardCornerDetector::flipLine(line_float_t line) {
  float temp = line.startx;
  line.startx = line.endx;
  line.endx = temp;
  temp = line.starty;
  line.starty = line.endy;
  line.endy = temp;
  return line;
}

float CardCornerDetector::distance(point_t p1, point_t p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

pair<vector<point_t>, int>
CardCornerDetector::getCorners(unsigned char *frameByteArray, int frameWidth,
                               int frameHeight, int guideFinderWidth,
                               int guideFinderHeight) {

  auto grayscaled = Utils::grayscale(frameByteArray, frameWidth, frameHeight);

  // initialize lines vector and bounding box
  vector<line_float_t> lines;
  boundingbox_t bbox;
  bbox.x = 0;
  bbox.y = 0;
  bbox.width = frameWidth;
  bbox.height = frameHeight;

  // calculate scale factor
  float scale = params.resizedWidth / (float)frameWidth;

  // use houghlines to detect lines
  HoughLineDetector(
      grayscaled, frameWidth, frameHeight, scale, scale, params.sigma,
      params.cannyLowerThreshold, params.cannyUpperThreshold, 1, PI / 180,
      params.houghlineMinLineLengthRatio * params.resizedWidth,
      params.houghlineMaxLineGapRatio * params.resizedWidth,
      params.houghlineThreshold, HOUGH_LINE_PROBABILISTIC, bbox, lines);

  // only keep the lines in the detection regions (top, bottom, left, right)
  int detectionArea = frameWidth * params.detectionAreaRatio;
  vector<line_float_t> linesTop, linesBottom, linesLeft, linesRight;

  for (int i = 0; i < lines.size(); i++) {
    line_float_t line = lines[i];
    if (line.startx < detectionArea && line.endx < detectionArea) {
      linesLeft.push_back(line);
    } else if (line.startx > frameWidth - detectionArea &&
               line.endx > frameWidth - detectionArea) {
      linesRight.push_back(line);
    } else if (line.starty < detectionArea && line.endy < detectionArea) {
      linesTop.push_back(line);
    } else if (line.starty > frameHeight - detectionArea &&
               line.endy > frameHeight - detectionArea) {
      linesBottom.push_back(line);
    }
  }

  // combine the region lines
  vector<line_float_t> foundLines;
  foundLines.insert(foundLines.end(), linesTop.begin(), linesTop.end());
  foundLines.insert(foundLines.end(), linesBottom.begin(), linesBottom.end());
  foundLines.insert(foundLines.end(), linesLeft.begin(), linesLeft.end());
  foundLines.insert(foundLines.end(), linesRight.begin(), linesRight.end());

  // check in all 4 corners for points that are within the detection area
  // and save the running average of the x and y coordinates
  // also orientate lines so they go from left to right and top to bottom
  point_t pointTopStart, pointTopEnd, pointBottomStart, pointBottomEnd,
      pointLeftStart, pointLeftEnd, pointRightStart, pointRightEnd;
  int numTopStart = 0, numTopEnd = 0, numBottomStart = 0, numBottomEnd = 0,
      numLeftStart = 0, numLeftEnd = 0, numRightStart = 0, numRightEnd = 0;

  for (int i = 0; i < linesTop.size(); i++) {
    line_float_t line = linesTop[i];
    if (line.startx > line.endx)
      line = flipLine(line);
    if (line.startx <= detectionArea && line.starty <= detectionArea) {
      if (numTopStart == 0) {
        pointTopStart = {(int)line.startx, (int)line.starty, 0};
      } else {
        pointTopStart.x =
            (pointTopStart.x * numTopStart + line.startx) / (numTopStart + 1);
        pointTopStart.y =
            (pointTopStart.y * numTopStart + line.starty) / (numTopStart + 1);
      }
      numTopStart++;
    }
    if (line.endx >= frameWidth - detectionArea && line.endy <= detectionArea) {
      if (numTopEnd == 0) {
        pointTopEnd = {(int)line.endx, (int)line.endy, 0};
      } else {
        pointTopEnd.x =
            (pointTopEnd.x * numTopEnd + line.endx) / (numTopEnd + 1);
        pointTopEnd.y =
            (pointTopEnd.y * numTopEnd + line.endy) / (numTopEnd + 1);
      }
      numTopEnd++;
    }
  }

  for (int i = 0; i < linesBottom.size(); i++) {
    line_float_t line = linesBottom[i];
    if (line.startx > line.endx)
      line = flipLine(line);
    if (line.startx <= detectionArea &&
        line.starty >= frameHeight - detectionArea) {
      if (numBottomStart == 0) {
        pointBottomStart = {(int)line.startx, (int)line.starty, 0};
      } else {
        pointBottomStart.x =
            (pointBottomStart.x * numBottomStart + line.startx) /
            (numBottomStart + 1);
        pointBottomStart.y =
            (pointBottomStart.y * numBottomStart + line.starty) /
            (numBottomStart + 1);
      }
      numBottomStart++;
    }
    if (line.endx >= frameWidth - detectionArea &&
        line.endy >= frameHeight - detectionArea) {
      if (numBottomEnd == 0) {
        pointBottomEnd = {(int)line.endx, (int)line.endy, 0};
      } else {
        pointBottomEnd.x =
            (pointBottomEnd.x * numBottomEnd + line.endx) / (numBottomEnd + 1);
        pointBottomEnd.y =
            (pointBottomEnd.y * numBottomEnd + line.endy) / (numBottomEnd + 1);
      }
      numBottomEnd++;
    }
  }

  for (int i = 0; i < linesLeft.size(); i++) {
    line_float_t line = linesLeft[i];
    if (line.starty > line.endy)
      line = flipLine(line);
    if (line.startx <= detectionArea && line.starty <= detectionArea) {
      if (numLeftStart == 0) {
        pointLeftStart = {(int)line.startx, (int)line.starty, 0};
      } else {
        pointLeftStart.x = (pointLeftStart.x * numLeftStart + line.startx) /
                           (numLeftStart + 1);
        pointLeftStart.y = (pointLeftStart.y * numLeftStart + line.starty) /
                           (numLeftStart + 1);
      }
      numLeftStart++;
    }
    if (line.endx <= detectionArea &&
        line.endy >= frameHeight - detectionArea) {
      if (numLeftEnd == 0) {
        pointLeftEnd = {(int)line.endx, (int)line.endy, 0};
      } else {
        pointLeftEnd.x =
            (pointLeftEnd.x * numLeftEnd + line.endx) / (numLeftEnd + 1);
        pointLeftEnd.y =
            (pointLeftEnd.y * numLeftEnd + line.endy) / (numLeftEnd + 1);
      }
      numLeftEnd++;
    }
  }

  for (int i = 0; i < linesRight.size(); i++) {
    line_float_t line = linesRight[i];
    if (line.starty > line.endy)
      line = flipLine(line);
    if (line.startx >= frameWidth - detectionArea &&
        line.starty <= detectionArea) {
      if (numRightStart == 0) {
        pointRightStart = {(int)line.startx, (int)line.starty, 0};
      } else {
        pointRightStart.x = (pointRightStart.x * numRightStart + line.startx) /
                            (numRightStart + 1);
        pointRightStart.y = (pointRightStart.y * numRightStart + line.starty) /
                            (numRightStart + 1);
      }
      numRightStart++;
    }
    if (line.endx >= frameWidth - detectionArea &&
        line.endy >= frameHeight - detectionArea) {
      if (numRightEnd == 0) {
        pointRightEnd = {(int)line.endx, (int)line.endy, 0};
      } else {
        pointRightEnd.x =
            (pointRightEnd.x * numRightEnd + line.endx) / (numRightEnd + 1);
        pointRightEnd.y =
            (pointRightEnd.y * numRightEnd + line.endy) / (numRightEnd + 1);
      }
      numRightEnd++;
    }
  }

  // store the corners of the frame and detection zones
  int guideFinderGapX = (frameWidth - guideFinderWidth) / 2;
  int guideFinderGapY = (frameHeight - guideFinderHeight) / 2;

  point_t guideFinderTopLeft = {guideFinderGapX, guideFinderGapY};
  point_t guideFinderTopRight = {frameWidth - guideFinderGapX, guideFinderGapY};
  point_t guideFinderBottomLeft = {guideFinderGapX,
                                   frameHeight - guideFinderGapY};
  point_t guideFinderBottomRight = {frameWidth - guideFinderGapX,
                                    frameHeight - guideFinderGapY};

  point_t detectionTopLeft = {detectionArea, detectionArea};
  point_t detectionTopRight = {frameWidth - detectionArea, detectionArea};
  point_t detectionBottomLeft = {detectionArea, frameHeight - detectionArea};
  point_t detectionBottomRight = {frameWidth - detectionArea,
                                  frameHeight - detectionArea};

  float frameToGuideDist = distance(point_t{0, 0}, guideFinderTopLeft);
  float detectionToGuideDist = distance(detectionTopLeft, guideFinderTopLeft);

  // the corner points are the average of the 2 edge points if they exist
  point_t cornerTopLeft, cornerTopRight, cornerBottomLeft, cornerBottomRight;
  int cornerCount = 0;

  if (numTopStart > 0 && numLeftStart > 0) {
    cornerCount++;
    cornerTopLeft.x = (int)((pointTopStart.x + pointLeftStart.x) / 2);
    cornerTopLeft.y = (int)((pointTopStart.y + pointLeftStart.y) / 2);
    if (cornerTopLeft.x < guideFinderTopLeft.x &&
        cornerTopLeft.y < guideFinderTopLeft.y) {
      cornerTopLeft.score =
          1 - distance(cornerTopLeft, guideFinderTopLeft) / frameToGuideDist;
    } else {
      cornerTopLeft.score = 1 - distance(cornerTopLeft, guideFinderTopLeft) /
                                    detectionToGuideDist;
    }
  } else {
    cornerTopLeft = {-1, -1, 0};
  }

  if (numTopEnd > 0 && numRightStart > 0) {
    cornerCount++;
    cornerTopRight.x = (int)((pointTopEnd.x + pointRightStart.x) / 2);
    cornerTopRight.y = (int)((pointTopEnd.y + pointRightStart.y) / 2);
    if (cornerTopRight.x > guideFinderTopRight.x &&
        cornerTopRight.y < guideFinderTopRight.y) {
      cornerTopRight.score =
          1 - distance(cornerTopRight, guideFinderTopRight) / frameToGuideDist;
    } else {
      cornerTopRight.score = 1 - distance(cornerTopRight, guideFinderTopRight) /
                                     detectionToGuideDist;
    }
  } else {
    cornerTopRight = {-1, -1, 0};
  }

  if (numBottomStart > 0 && numLeftEnd > 0) {
    cornerCount++;
    cornerBottomLeft.x = (int)((pointBottomStart.x + pointLeftEnd.x) / 2);
    cornerBottomLeft.y = (int)((pointBottomStart.y + pointLeftEnd.y) / 2);
    if (cornerBottomLeft.x < guideFinderBottomLeft.x &&
        cornerBottomLeft.y > guideFinderBottomLeft.y) {
      cornerBottomLeft.score =
          1 -
          distance(cornerBottomLeft, guideFinderBottomLeft) / frameToGuideDist;
    } else {
      cornerBottomLeft.score =
          1 - distance(cornerBottomLeft, guideFinderBottomLeft) /
                  detectionToGuideDist;
    }
  } else {
    cornerBottomLeft = {-1, -1, 0};
  }

  if (numBottomEnd > 0 && numRightEnd > 0) {
    cornerCount++;
    cornerBottomRight.x = (int)((pointBottomEnd.x + pointRightEnd.x) / 2);
    cornerBottomRight.y = (int)((pointBottomEnd.y + pointRightEnd.y) / 2);
    if (cornerBottomRight.x > guideFinderBottomRight.x &&
        cornerBottomRight.y > guideFinderBottomRight.y) {
      cornerBottomRight.score =
          1 - distance(cornerBottomRight, guideFinderBottomRight) /
                  frameToGuideDist;
    } else {
      cornerBottomRight.score =
          1 - distance(cornerBottomRight, guideFinderBottomRight) /
                  detectionToGuideDist;
    }
  } else {
    cornerBottomRight = {-1, -1, 0};
  }

  // // append the corner count and corner coordinates to an output int vector
  // vector<int> foundCorners;

  // foundCorners.push_back(cornerCount);
  // foundCorners.push_back(cornerTopLeft.x);
  // foundCorners.push_back(cornerTopLeft.y);
  // foundCorners.push_back(cornerTopRight.x);
  // foundCorners.push_back(cornerTopRight.y);
  // foundCorners.push_back(cornerBottomLeft.x);
  // foundCorners.push_back(cornerBottomLeft.y);
  // foundCorners.push_back(cornerBottomRight.x);
  // foundCorners.push_back(cornerBottomRight.y);

  // calculate the average score of the corners
  // float cornerScore = (cornerTopLeft.score + cornerTopRight.score +
  //                      cornerBottomLeft.score + cornerBottomRight.score) /
  //                     4;

  //// --- START OF DRAWINGS --- ////
  // draw the found lines on a blank image
  unsigned char *frameByteArrayOut =
      new unsigned char[frameWidth * frameHeight];
  for (int i = 0; i < frameWidth * frameHeight; i++) {
    frameByteArrayOut[i] = 0;
  }
  for (int i = 0; i < lines.size(); i++) {
    int x1 = lines[i].startx;
    int y1 = lines[i].starty;
    int x2 = lines[i].endx;
    int y2 = lines[i].endy;
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    while (true) {
      frameByteArrayOut[y1 * frameWidth + x1] = 255;
      if (x1 == x2 && y1 == y2) {
        break;
      }
      int e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x1 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y1 += sy;
      }
    }
  }

  // draw the four corners
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int x = cornerTopLeft.x;
      int y = cornerTopLeft.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = cornerTopRight.x;
      y = cornerTopRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = cornerBottomLeft.x;
      y = cornerBottomLeft.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = cornerBottomRight.x;
      y = cornerBottomRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
    }
  }

  // draw the guide finder corners
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int x = guideFinderTopLeft.x;
      int y = guideFinderTopLeft.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 125;
      }
      x = guideFinderTopRight.x;
      y = guideFinderTopRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 125;
      }
      x = guideFinderBottomLeft.x;
      y = guideFinderBottomLeft.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 125;
      }
      x = guideFinderBottomRight.x;
      y = guideFinderBottomRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 125;
      }
    }
  }

  // draw the detection corners
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      int x = detectionTopLeft.x;
      int y = detectionTopLeft.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 125;
      }
      x = detectionTopRight.x;
      y = detectionTopRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 125;
      }
      x = detectionBottomLeft.x;
      y = detectionBottomLeft.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 125;
      }
      x = detectionBottomRight.x;
      y = detectionBottomRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 125;
      }
    }
  }

  // convert the unsigned char array to an opencv mat for display
  cv::Mat frameMat(frameHeight, frameWidth, CV_8UC1, frameByteArrayOut);
  imshow("corners", frameMat);
  //// --- END OF DRAWINGS --- ////

  // return the vector of found corners
  delete grayscaled;
  return make_pair(vector<point_t>{cornerTopLeft, cornerTopRight,
                                   cornerBottomLeft, cornerBottomRight},
                   cornerCount);
  // return make_pair(foundCorners, (int)(cornerScore * 100));
}