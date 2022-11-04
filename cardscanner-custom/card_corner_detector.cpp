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
  float detectionArea = frameWidth * params.detectionAreaRatio;
  vector<line_float_t> linesTop;
  vector<line_float_t> linesBottom;
  vector<line_float_t> linesLeft;
  vector<line_float_t> linesRight;

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

  // orientate lines so they go from left to right and top to bottom
  for (int i = 0; i < linesTop.size(); i++)
    if (linesTop[i].startx > linesTop[i].endx)
      linesTop[i] = flipLine(linesTop[i]);
  for (int i = 0; i < linesBottom.size(); i++)
    if (linesBottom[i].startx > linesBottom[i].endx)
      linesBottom[i] = flipLine(linesBottom[i]);
  for (int i = 0; i < linesLeft.size(); i++)
    if (linesLeft[i].starty > linesLeft[i].endy)
      linesLeft[i] = flipLine(linesLeft[i]);
  for (int i = 0; i < linesRight.size(); i++)
    if (linesRight[i].starty > linesRight[i].endy)
      linesRight[i] = flipLine(linesRight[i]);

  // combine the region lines
  vector<line_float_t> foundLines;
  foundLines.insert(foundLines.end(), linesTop.begin(), linesTop.end());
  foundLines.insert(foundLines.end(), linesBottom.begin(), linesBottom.end());
  foundLines.insert(foundLines.end(), linesLeft.begin(), linesLeft.end());
  foundLines.insert(foundLines.end(), linesRight.begin(), linesRight.end());

  // check in all 4 corners if there is a line with x and y coordinates
  // within detection area ratio multiplied by the image width and height
  vector<point_t> cornersTopLeft, cornersTopRight, cornersBottomLeft,
      cornersBottomRight;

  for (int i = 0; i < linesTop.size(); i++) {
    line_float_t line = linesTop[i];
    if (line.startx <= detectionArea && line.starty <= detectionArea)
      cornersTopLeft.push_back(point_t{(int)line.startx, (int)line.starty});
    if (line.endx >= frameWidth - detectionArea && line.endy <= detectionArea)
      cornersTopRight.push_back(point_t{(int)line.endx, (int)line.endy});
  }
  for (int i = 0; i < linesBottom.size(); i++) {
    line_float_t line = linesBottom[i];
    if (line.startx <= detectionArea &&
        line.starty >= frameHeight - detectionArea)
      cornersBottomLeft.push_back(point_t{(int)line.startx, (int)line.starty});
    if (line.endx >= frameWidth - detectionArea &&
        line.endy >= frameHeight - detectionArea)
      cornersBottomRight.push_back(point_t{(int)line.endx, (int)line.endy});
  }
  for (int i = 0; i < linesLeft.size(); i++) {
    line_float_t line = linesLeft[i];
    if (line.startx <= detectionArea && line.starty <= detectionArea)
      cornersTopLeft.push_back(point_t{(int)line.startx, (int)line.starty});
    if (line.endx <= detectionArea && line.endy >= frameHeight - detectionArea)
      cornersBottomLeft.push_back(point_t{(int)line.endx, (int)line.endy});
  }
  for (int i = 0; i < linesRight.size(); i++) {
    line_float_t line = linesRight[i];
    if (line.startx >= frameWidth - detectionArea &&
        line.starty <= detectionArea)
      cornersTopRight.push_back(point_t{(int)line.startx, (int)line.starty});
    if (line.endx >= frameWidth - detectionArea &&
        line.endy >= frameHeight - detectionArea)
      cornersBottomRight.push_back(point_t{(int)line.endx, (int)line.endy});
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

  point_t detectionTopLeft = {(int)(detectionArea), (int)(detectionArea)};
  point_t detectionTopRight = {(int)(frameWidth - detectionArea),
                               (int)(detectionArea)};
  point_t detectionBottomLeft = {(int)(detectionArea),
                                 (int)(frameHeight - detectionArea)};
  point_t detectionBottomRight = {(int)(frameWidth - detectionArea),
                                  (int)(frameHeight - detectionArea)};

  float frameToGuideDist = distance(point_t{0, 0}, guideFinderTopLeft);
  float detectionToGuideDist = distance(detectionTopLeft, guideFinderTopLeft);

  // save the furthest x and y coordinates from the four corners and
  // calculate score for how far away the corners are from the guide finder
  point_t cornerTopLeft, cornerTopRight, cornerBottomLeft, cornerBottomRight;
  int cornerCount = 0;

  if (cornersTopLeft.size() > 0) {
    cornerCount++;
    cornerTopLeft = cornersTopLeft[0];
    for (int i = 0; i < cornersTopLeft.size(); i++) {
      cornerTopLeft.x = min(cornerTopLeft.x, cornersTopLeft[i].x);
      cornerTopLeft.y = min(cornerTopLeft.y, cornersTopLeft[i].y);
    }
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

  if (cornersTopRight.size() > 0) {
    cornerCount++;
    cornerTopRight = cornersTopRight[0];
    for (int i = 0; i < cornersTopRight.size(); i++) {
      cornerTopRight.x = max(cornerTopRight.x, cornersTopRight[i].x);
      cornerTopRight.y = min(cornerTopRight.y, cornersTopRight[i].y);
    }
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

  if (cornersBottomLeft.size() > 0) {
    cornerCount++;
    cornerBottomLeft = cornersBottomLeft[0];
    for (int i = 0; i < cornersBottomLeft.size(); i++) {
      cornerBottomLeft.x = min(cornerBottomLeft.x, cornersBottomLeft[i].x);
      cornerBottomLeft.y = max(cornerBottomLeft.y, cornersBottomLeft[i].y);
    }
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

  if (cornersBottomRight.size() > 0) {
    cornerCount++;
    cornerBottomRight = cornersBottomRight[0];
    for (int i = 0; i < cornersBottomRight.size(); i++) {
      cornerBottomRight.x = max(cornerBottomRight.x, cornersBottomRight[i].x);
      cornerBottomRight.y = max(cornerBottomRight.y, cornersBottomRight[i].y);
    }
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

  // append the corner count and corner coordinates to an output int vector
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

  // return the vector of found corners
  delete grayscaled;
  return make_pair(vector<point_t>{cornerTopLeft, cornerTopRight,
                                   cornerBottomLeft, cornerBottomRight},
                   cornerCount);
  // return make_pair(foundCorners, (int)(cornerScore * 100));
}