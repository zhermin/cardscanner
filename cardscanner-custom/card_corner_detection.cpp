#include "card_corner_detector.h"
#include "houghlines/houghlines.h"

#include <android/log.h>
#define TAG "AUTO_CAPTURE_PROCESSOR"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using namespace std;

struct {
  float resizedWidth =
      480; // new width of sized down image for faster processing
  float detectionAreaRatio =
      0.15; // ratio of the detection area to the image area (30)
  int cannyLowerThreshold =
      60; // rejected if pixel gradient is below lower threshold
  int cannyUpperThreshold =
      180; // accepted if pixel gradient is above upper threshold
  int houghlineThreshold = 30; // minimum intersections to detect a line
  float houghlineMinLineLengthRatio =
      0.1; // minimum length of a line to detect (30)
  float houghlineMaxLineGapRatio =
      0.1; // maximum gap between two potential lines to join into 1 line (30)
} PARAMS;

unsigned char *grayscale(const unsigned char *frame, int frameWidth,
                         int frameHeight) {
  // convert the 4-channel RGBA image to a single channel grayscale image
  unsigned char *grayscaled = new unsigned char[frameWidth * frameHeight];
  for (int i = 0; i < frameWidth * frameHeight; i++) {
    float r = frame[i * 4];
    float g = frame[i * 4 + 1];
    float b = frame[i * 4 + 2];
    grayscaled[i] = (unsigned char)(int)(0.299 * r + 0.587 * g + 0.114 * b);
  }

  return grayscaled;
}

line_float_t flipLine(line_float_t line) {
  float temp = line.startx;
  line.startx = line.endx;
  line.endx = temp;
  temp = line.starty;
  line.starty = line.endy;
  line.endy = temp;
  return line;
}

float distance(point_t p1, point_t p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

vector<point_t> getCorners(unsigned char *frameByteArray, int frameWidth,
                           int frameHeight, int guideFinderWidth,
                           int guideFinderHeight) {

  auto grayscaled = grayscale(frameByteArray, frameWidth, frameHeight);

  // initialize lines vector and bounding box
  vector<line_float_t> lines;
  boundingbox_t bbox;
  bbox.x = 0;
  bbox.y = 0;
  bbox.width = frameWidth;
  bbox.height = frameHeight;

  // calculate scale factor
  float scale = PARAMS.resizedWidth / (float)frameWidth;

  // use houghlines to detect lines
  HoughLineDetector(
      frameByteArray, frameWidth, frameHeight, scale, scale,
      PARAMS.cannyLowerThreshold, PARAMS.cannyUpperThreshold, 1, PI / 180,
      PARAMS.houghlineMinLineLengthRatio * PARAMS.resizedWidth,
      PARAMS.houghlineMaxLineGapRatio * PARAMS.resizedWidth,
      PARAMS.houghlineThreshold, HOUGH_LINE_PROBABILISTIC, bbox, lines);

  // only keep the lines in the detection regions (top, bottom, left, right)
  vector<line_float_t> linesTop;
  vector<line_float_t> linesBottom;
  vector<line_float_t> linesLeft;
  vector<line_float_t> linesRight;
  for (int i = 0; i < lines.size(); i++) {
    line_float_t line = lines[i];
    // if lines both x or y are 1 pixel away from edges, reject as may be noise
    if (line.startx == 1 && line.endx == 1 ||
        line.starty == 1 && line.endy == 1 ||
        line.startx == frameWidth - 1 && line.endx == frameWidth - 1 ||
        line.starty == frameHeight - 1 && line.endy == frameHeight - 1)
      continue;

    if (line.startx < frameWidth * PARAMS.detectionAreaRatio &&
        line.endx < frameWidth * PARAMS.detectionAreaRatio) {
      linesLeft.push_back(line);
    } else if (line.startx > frameWidth * (1 - PARAMS.detectionAreaRatio) &&
               line.endx > frameWidth * (1 - PARAMS.detectionAreaRatio)) {
      linesRight.push_back(line);
    } else if (line.starty < frameHeight * PARAMS.detectionAreaRatio &&
               line.endy < frameHeight * PARAMS.detectionAreaRatio) {
      linesTop.push_back(line);
    } else if (line.starty > frameHeight * (1 - PARAMS.detectionAreaRatio) &&
               line.endy > frameHeight * (1 - PARAMS.detectionAreaRatio)) {
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
  for (int i = 0; i < foundLines.size(); i++) {
    line_float_t line = foundLines[i];
    int minx = min(line.startx, line.endx);
    int miny = min(line.starty, line.endy);
    int maxx = max(line.startx, line.endx);
    int maxy = max(line.starty, line.endy);
    if (minx <= frameWidth * PARAMS.detectionAreaRatio &&
        miny <= frameHeight * PARAMS.detectionAreaRatio) {
      cornersTopLeft.push_back(point_t{minx, miny});
    }
    if (maxx >= frameWidth * (1 - PARAMS.detectionAreaRatio) &&
        miny <= frameHeight * PARAMS.detectionAreaRatio) {
      cornersTopRight.push_back(point_t{maxx, miny});
    }
    if (minx <= frameWidth * PARAMS.detectionAreaRatio &&
        maxy >= frameHeight * (1 - PARAMS.detectionAreaRatio)) {
      cornersBottomLeft.push_back(point_t{minx, maxy});
    }
    if (maxx >= frameWidth * (1 - PARAMS.detectionAreaRatio) &&
        maxy >= frameHeight * (1 - PARAMS.detectionAreaRatio)) {
      cornersBottomRight.push_back(point_t{maxx, maxy});
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

  point_t detectionTopLeft = {(int)(frameWidth * PARAMS.detectionAreaRatio),
                              (int)(frameHeight * PARAMS.detectionAreaRatio)};
  point_t detectionTopRight = {
      (int)(frameWidth * (1 - PARAMS.detectionAreaRatio)),
      (int)(frameHeight * PARAMS.detectionAreaRatio)};
  point_t detectionBottomLeft = {
      (int)(frameWidth * PARAMS.detectionAreaRatio),
      (int)(frameHeight * (1 - PARAMS.detectionAreaRatio))};
  point_t detectionBottomRight = {
      (int)(frameWidth * (1 - PARAMS.detectionAreaRatio)),
      (int)(frameHeight * (1 - PARAMS.detectionAreaRatio))};

  float frameToGuideDist = distance(point_t{0, 0}, guideFinderTopLeft);
  float detectionToGuideDist = distance(detectionTopLeft, guideFinderTopLeft);

  // save the furthest x and y coordinates from the four corners and
  // calculate score for how far away the corners are from the guide finder
  point_t cornerTopLeft, cornerTopRight, cornerBottomLeft, cornerBottomRight;

  if (cornersTopLeft.size() > 0) {
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

  // return the vector of found corners
  return {cornerTopLeft, cornerTopRight, cornerBottomLeft, cornerBottomRight};
}