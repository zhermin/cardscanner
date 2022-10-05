#include "houghlines/houghlines.h"
#include <fstream>

char type[10];
int width;
int height;
int intensity;

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

int main(int argc, char **argv) {
  // USAGE
  // $ ./main.o <pgm image (in ./assets folder)> <0/1 Standard/Probabilistic>
  // EXAMPLE
  // $ ./main.o card5 1; echo $?
  // >> -1 (error) or 0-4 (number of found corners)

  if (argc < 3)
    return -1;

  std::ifstream inFile("assets/" + std::string(argv[1]) + ".pgm",
                       std::ios::binary);
  if (!inFile.is_open())
    return -1;

  // read header (.pgm type and intensity header info are unused)
  inFile >> type >> width >> height >> intensity;
  unsigned char *frameByteArray = new unsigned char[width * height];
  inFile.read((char *)frameByteArray, width * height);
  inFile.close();

  // initialize lines vector and bounding box
  std::vector<line_float_t> lines;
  boundingbox_t bbox;
  bbox.x = 0;
  bbox.y = 0;
  bbox.width = width;
  bbox.height = height;

  // calculate scale factor
  float scale = PARAMS.resizedWidth / (float)width;

  // use houghlines to detect lines
  if (std::stoi(argv[2]) == 0) {
    HoughLineDetector(frameByteArray, width, height, scale, scale,
                      PARAMS.cannyLowerThresholdRatio * PARAMS.resizedWidth,
                      PARAMS.cannyUpperThresholdRatio * PARAMS.resizedWidth, 1,
                      PI / 180, 0, PI,
                      PARAMS.houghlineThresholdRatio * PARAMS.resizedWidth,
                      HOUGH_LINE_STANDARD, bbox, lines);
  } else {
    HoughLineDetector(frameByteArray, width, height, scale, scale,
                      PARAMS.cannyLowerThresholdRatio * PARAMS.resizedWidth,
                      PARAMS.cannyUpperThresholdRatio * PARAMS.resizedWidth, 1,
                      PI / 180,
                      PARAMS.houghlineMinLineLengthRatio * PARAMS.resizedWidth,
                      PARAMS.houghlineMaxLineGapRatio * PARAMS.resizedWidth,
                      PARAMS.houghlineThresholdRatio * PARAMS.resizedWidth,
                      HOUGH_LINE_PROBABILISTIC, bbox, lines);
  }

  // only keep the lines in the detection regions (top, bottom, left, right)
  std::vector<line_float_t> linesTop;
  std::vector<line_float_t> linesBottom;
  std::vector<line_float_t> linesLeft;
  std::vector<line_float_t> linesRight;
  for (int i = 0; i < lines.size(); i++) {
    line_float_t line = lines[i];
    if (line.startx == 1 && line.endx == 1)
      continue;
    if (line.startx < width * PARAMS.areaDetectionRatio &&
        line.endx < width * PARAMS.areaDetectionRatio) {
      linesLeft.push_back(line);
    } else if (line.startx > width * (1 - PARAMS.areaDetectionRatio) &&
               line.endx > width * (1 - PARAMS.areaDetectionRatio)) {
      linesRight.push_back(line);
    } else if (line.starty < height * PARAMS.areaDetectionRatio &&
               line.endy < height * PARAMS.areaDetectionRatio) {
      linesTop.push_back(line);
    } else if (line.starty > height * (1 - PARAMS.areaDetectionRatio) &&
               line.endy > height * (1 - PARAMS.areaDetectionRatio)) {
      linesBottom.push_back(line);
    }
  }

  // combine the region lines
  std::vector<line_float_t> foundLines;
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
    int minx = std::min(line.startx, line.endx);
    int miny = std::min(line.starty, line.endy);
    int maxx = std::max(line.startx, line.endx);
    int maxy = std::max(line.starty, line.endy);
    if (minx < width * PARAMS.areaDetectionRatio &&
        miny < height * PARAMS.areaDetectionRatio) {
      cornerTopLeft.x = minx;
      cornerTopLeft.y = miny;
      foundTopLeft = true;
    } else if (maxx > width * (1 - PARAMS.areaDetectionRatio) &&
               miny < height * PARAMS.areaDetectionRatio) {
      cornerTopRight.x = maxx;
      cornerTopRight.y = miny;
      foundTopRight = true;
    } else if (minx < width * PARAMS.areaDetectionRatio &&
               maxy > height * (1 - PARAMS.areaDetectionRatio)) {
      cornerBottomLeft.x = minx;
      cornerBottomLeft.y = maxy;
      foundBottomLeft = true;
    } else if (maxx > width * (1 - PARAMS.areaDetectionRatio) &&
               maxy > height * (1 - PARAMS.areaDetectionRatio)) {
      cornerBottomRight.x = maxx;
      cornerBottomRight.y = maxy;
      foundBottomRight = true;
    }
  }

  // draw the found lines on a blank image
  unsigned char *frameByteArrayOut = new unsigned char[width * height];
  for (int i = 0; i < width * height; i++) {
    frameByteArrayOut[i] = 0;
  }
  for (int i = 0; i < foundLines.size(); i++) {
    int x1 = foundLines[i].startx;
    int y1 = foundLines[i].starty;
    int x2 = foundLines[i].endx;
    int y2 = foundLines[i].endy;
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;
    while (true) {
      frameByteArrayOut[y1 * width + x1] = 255;
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

  // draw the found corners on the image
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      if (foundTopLeft) {
        frameByteArrayOut[(cornerTopLeft.y - 2 + i) * width +
                          (cornerTopLeft.x - 2 + j)] = 255;
      }
      if (foundTopRight) {
        frameByteArrayOut[(cornerTopRight.y - 2 + i) * width +
                          (cornerTopRight.x - 2 + j)] = 255;
      }
      if (foundBottomLeft) {
        frameByteArrayOut[(cornerBottomLeft.y - 2 + i) * width +
                          (cornerBottomLeft.x - 2 + j)] = 255;
      }
      if (foundBottomRight) {
        frameByteArrayOut[(cornerBottomRight.y - 2 + i) * width +
                          (cornerBottomRight.x - 2 + j)] = 255;
      }
    }
  }

  // write image to a new pgm file with the same header as the input file
  std::ofstream outFile("assets/outputs/" + std::string(argv[1]) + "_out" +
                            std::string(argv[2]) + ".pgm",
                        std::ios::binary);

  outFile << type << std::endl
          << width << " " << height << std::endl
          << intensity << std::endl;
  outFile.write((char *)frameByteArrayOut, width * height);
  outFile.close();

  // free memory
  delete[] frameByteArray;
  delete[] frameByteArrayOut;

  // if 3 or more corners are found, a card is detected
  int foundCorners = int(foundTopLeft) + int(foundTopRight) +
                     int(foundBottomLeft) + int(foundBottomRight);

  // return the number of found corners
  // note: the coordinates of the corners are in corner*** .x and .y members
  return foundCorners;
}