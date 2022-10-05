#include "houghlines/houghlines.h"

using namespace std;

struct {
  float resizedWidth =
      300; // new width of sized down image for faster processing
  float areaDetectionRatio =
      0.10; // ratio of the detection area to the image area (30)
  float cannyLowerThresholdRatio =
      0.2; // rejected if pixel gradient is below lower threshold (60)
  float cannyUpperThresholdRatio =
      0.6; // accepted if pixel gradient is above upper threshold (180)
  float houghlineThresholdRatio =
      0.1; // minimum intersections to detect a line (30)
  float houghlineMinLineLengthRatio =
      0.1; // minimum length of a line to detect (30)
  float houghlineMaxLineGapRatio =
      0.1; // maximum gap between two potential lines to join into 1 line (30)
} PARAMS;

vector<point_t> getCorners(unsigned char *frameByteArray, int frameWidth,
                           int frameHeight) {
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
  HoughLineDetector(frameByteArray, frameWidth, frameHeight, scale, scale,
                    PARAMS.cannyLowerThresholdRatio * PARAMS.resizedWidth,
                    PARAMS.cannyUpperThresholdRatio * PARAMS.resizedWidth, 1,
                    PI / 180,
                    PARAMS.houghlineMinLineLengthRatio * PARAMS.resizedWidth,
                    PARAMS.houghlineMaxLineGapRatio * PARAMS.resizedWidth,
                    PARAMS.houghlineThresholdRatio * PARAMS.resizedWidth,
                    HOUGH_LINE_PROBABILISTIC, bbox, lines);

  // only keep the lines in the detection regions (top, bottom, left, right)
  vector<line_float_t> linesTop;
  vector<line_float_t> linesBottom;
  vector<line_float_t> linesLeft;
  vector<line_float_t> linesRight;
  for (int i = 0; i < lines.size(); i++) {
    line_float_t line = lines[i];
    if (line.startx == 1 && line.endx == 1)
      continue;
    if (line.startx < frameWidth * PARAMS.areaDetectionRatio &&
        line.endx < frameWidth * PARAMS.areaDetectionRatio) {
      linesLeft.push_back(line);
    } else if (line.startx > frameWidth * (1 - PARAMS.areaDetectionRatio) &&
               line.endx > frameWidth * (1 - PARAMS.areaDetectionRatio)) {
      linesRight.push_back(line);
    } else if (line.starty < frameHeight * PARAMS.areaDetectionRatio &&
               line.endy < frameHeight * PARAMS.areaDetectionRatio) {
      linesTop.push_back(line);
    } else if (line.starty > frameHeight * (1 - PARAMS.areaDetectionRatio) &&
               line.endy > frameHeight * (1 - PARAMS.areaDetectionRatio)) {
      linesBottom.push_back(line);
    }
  }

  // combine the region lines
  vector<line_float_t> foundLines;
  foundLines.insert(foundLines.end(), linesTop.begin(), linesTop.end());
  foundLines.insert(foundLines.end(), linesBottom.begin(), linesBottom.end());
  foundLines.insert(foundLines.end(), linesLeft.begin(), linesLeft.end());
  foundLines.insert(foundLines.end(), linesRight.begin(), linesRight.end());

  // check in all 4 corners if there is a line with x and y coordinates
  // within detection area ratio multiplied by the image width and height
  point_t cornerTopLeft, cornerTopRight, cornerBottomLeft, cornerBottomRight;
  bool foundTopLeft = false;
  bool foundTopRight = false;
  bool foundBottomLeft = false;
  bool foundBottomRight = false;
  for (int i = 0; i < foundLines.size(); i++) {
    line_float_t line = foundLines[i];
    int minx = min(line.startx, line.endx);
    int miny = min(line.starty, line.endy);
    int maxx = max(line.startx, line.endx);
    int maxy = max(line.starty, line.endy);
    if (minx < frameWidth * PARAMS.areaDetectionRatio &&
        miny < frameHeight * PARAMS.areaDetectionRatio) {
      cornerTopLeft.x = minx;
      cornerTopLeft.y = miny;
      foundTopLeft = true;
    } else if (maxx > frameWidth * (1 - PARAMS.areaDetectionRatio) &&
               miny < frameHeight * PARAMS.areaDetectionRatio) {
      cornerTopRight.x = maxx;
      cornerTopRight.y = miny;
      foundTopRight = true;
    } else if (minx < frameWidth * PARAMS.areaDetectionRatio &&
               maxy > frameHeight * (1 - PARAMS.areaDetectionRatio)) {
      cornerBottomLeft.x = minx;
      cornerBottomLeft.y = maxy;
      foundBottomLeft = true;
    } else if (maxx > frameWidth * (1 - PARAMS.areaDetectionRatio) &&
               maxy > frameHeight * (1 - PARAMS.areaDetectionRatio)) {
      cornerBottomRight.x = maxx;
      cornerBottomRight.y = maxy;
      foundBottomRight = true;
    }
  }

  // add the corners into a vector if they are found
  vector<point_t> foundCorners;
  if (foundTopLeft)
    foundCorners.push_back(cornerTopLeft);
  if (foundTopRight)
    foundCorners.push_back(cornerTopRight);
  if (foundBottomLeft)
    foundCorners.push_back(cornerBottomLeft);
  if (foundBottomRight)
    foundCorners.push_back(cornerBottomRight);

  // return the vector of found corners
  return foundCorners;
}