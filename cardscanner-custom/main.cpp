#include "houghlines/houghlines.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>

bool DEBUG = 1;
int experiments = 1;

char type[10];
int width;
int height;
int intensity;

// note: hardcoded guided viewfinder dimensions from SDK frontend
int guideFinderWidth = 440;
int guideFinderHeight = 277;

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

std::vector<point_t> getCorners(std::vector<float> &runtimes,
                                unsigned char *frameByteArray, int frameWidth,
                                int frameHeight, std::string filename,
                                int mode) {

  // note: frameWidth and frameHeight should be 480x301 when implemented
  if (frameWidth != 480 || frameHeight != 301) {
    std::cout << "Error: frame dimensions (" << frameWidth << "x" << frameHeight
              << ") are not 480x301" << std::endl;
    exit(1);
  }

  // start time
  auto start = std::chrono::system_clock::now();

  // initialize lines vector and bounding box
  std::vector<line_float_t> lines;
  boundingbox_t bbox;
  bbox.x = 0;
  bbox.y = 0;
  bbox.width = frameWidth;
  bbox.height = frameHeight;

  // calculate scale factor
  float scale = PARAMS.resizedWidth / (float)frameWidth;

  // use houghlines to detect lines
  if (mode == 0) {
    HoughLineDetector(frameByteArray, frameWidth, frameHeight, scale, scale,
                      PARAMS.cannyLowerThreshold, PARAMS.cannyUpperThreshold, 1,
                      PI / 180, 0, PI, PARAMS.houghlineThreshold,
                      HOUGH_LINE_STANDARD, bbox, lines);
  } else {
    HoughLineDetector(
        frameByteArray, frameWidth, frameHeight, scale, scale,
        PARAMS.cannyLowerThreshold, PARAMS.cannyUpperThreshold, 1, PI / 180,
        PARAMS.houghlineMinLineLengthRatio * PARAMS.resizedWidth,
        PARAMS.houghlineMaxLineGapRatio * PARAMS.resizedWidth,
        PARAMS.houghlineThreshold, HOUGH_LINE_PROBABILISTIC, bbox, lines);
  }

  // only keep lines outside of the detection regions (top, bottom, left, right)
  std::vector<line_float_t> linesTop;
  std::vector<line_float_t> linesBottom;
  std::vector<line_float_t> linesLeft;
  std::vector<line_float_t> linesRight;
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

  // print lines
  if (DEBUG) {
    std::cout << "Top lines: " << linesTop.size() << std::endl;
    for (int i = 0; i < linesTop.size(); i++) {
      line_float_t line = linesTop[i];
      std::cout << "Line " << i << ": " << line.startx << ", " << line.starty
                << " -> " << line.endx << ", " << line.endy << std::endl;
    }
    std::cout << "Bottom lines: " << linesBottom.size() << std::endl;
    for (int i = 0; i < linesBottom.size(); i++) {
      line_float_t line = linesBottom[i];
      std::cout << "Line " << i << ": " << line.startx << ", " << line.starty
                << " -> " << line.endx << ", " << line.endy << std::endl;
    }
    std::cout << "Left lines: " << linesLeft.size() << std::endl;
    for (int i = 0; i < linesLeft.size(); i++) {
      line_float_t line = linesLeft[i];
      std::cout << "Line " << i << ": " << line.startx << ", " << line.starty
                << " -> " << line.endx << ", " << line.endy << std::endl;
    }
    std::cout << "Right lines: " << linesRight.size() << std::endl;
    for (int i = 0; i < linesRight.size(); i++) {
      line_float_t line = linesRight[i];
      std::cout << "Line " << i << ": " << line.startx << ", " << line.starty
                << " -> " << line.endx << ", " << line.endy << std::endl;
    }
    std::cout << "---" << std::endl;
  }

  // combine the region lines
  std::vector<line_float_t> foundLines;
  foundLines.insert(foundLines.end(), linesTop.begin(), linesTop.end());
  foundLines.insert(foundLines.end(), linesBottom.begin(), linesBottom.end());
  foundLines.insert(foundLines.end(), linesLeft.begin(), linesLeft.end());
  foundLines.insert(foundLines.end(), linesRight.begin(), linesRight.end());

  // check in all 4 corners if there is a line with x and y coordinates
  // outside of detection area ratio multiplied by the image width and height
  std::vector<point_t> cornersTopLeft, cornersTopRight, cornersBottomLeft,
      cornersBottomRight;
  for (int i = 0; i < foundLines.size(); i++) {
    line_float_t line = foundLines[i];
    int minx = std::min(line.startx, line.endx);
    int miny = std::min(line.starty, line.endy);
    int maxx = std::max(line.startx, line.endx);
    int maxy = std::max(line.starty, line.endy);
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

  // print corners
  if (DEBUG) {
    std::cout << "Top left corners: " << cornersTopLeft.size() << std::endl;
    for (int i = 0; i < cornersTopLeft.size(); i++) {
      point_t point = cornersTopLeft[i];
      std::cout << "Corner " << i << ": " << point.x << ", " << point.y
                << std::endl;
    }
    std::cout << "Top right corners: " << cornersTopRight.size() << std::endl;
    for (int i = 0; i < cornersTopRight.size(); i++) {
      point_t point = cornersTopRight[i];
      std::cout << "Corner " << i << ": " << point.x << ", " << point.y
                << std::endl;
    }
    std::cout << "Bottom left corners: " << cornersBottomLeft.size()
              << std::endl;
    for (int i = 0; i < cornersBottomLeft.size(); i++) {
      point_t point = cornersBottomLeft[i];
      std::cout << "Corner " << i << ": " << point.x << ", " << point.y
                << std::endl;
    }
    std::cout << "Bottom right corners: " << cornersBottomRight.size()
              << std::endl;
    for (int i = 0; i < cornersBottomRight.size(); i++) {
      point_t point = cornersBottomRight[i];
      std::cout << "Corner " << i << ": " << point.x << ", " << point.y
                << std::endl;
    }
    std::cout << "---" << std::endl;
  }

  // draw the found lines on a blank image
  unsigned char *frameByteArrayOut =
      new unsigned char[frameWidth * frameHeight];
  for (int i = 0; i < frameWidth * frameHeight; i++) {
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
      cornerTopLeft.x = std::min(cornerTopLeft.x, cornersTopLeft[i].x);
      cornerTopLeft.y = std::min(cornerTopLeft.y, cornersTopLeft[i].y);
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
      cornerTopRight.x = std::max(cornerTopRight.x, cornersTopRight[i].x);
      cornerTopRight.y = std::min(cornerTopRight.y, cornersTopRight[i].y);
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
      cornerBottomLeft.x = std::min(cornerBottomLeft.x, cornersBottomLeft[i].x);
      cornerBottomLeft.y = std::max(cornerBottomLeft.y, cornersBottomLeft[i].y);
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
      cornerBottomRight.x =
          std::max(cornerBottomRight.x, cornersBottomRight[i].x);
      cornerBottomRight.y =
          std::max(cornerBottomRight.y, cornersBottomRight[i].y);
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

  float cornerScore = (cornerTopLeft.score + cornerTopRight.score +
                       cornerBottomLeft.score + cornerBottomRight.score) /
                      4;

  // -------------------------- END OF LOGIC -------------------------- //

  // end time
  auto end = std::chrono::system_clock::now();

  // runtime
  std::chrono::duration<double> elapsed_seconds = (end - start) * 1000;
  runtimes.push_back(elapsed_seconds.count());

  // print scores
  if (DEBUG) {
    printf("Top Left Score: %f\n", cornerTopLeft.score);
    printf("Top Right Score: %f\n", cornerTopRight.score);
    printf("Bottom Left Score: %f\n", cornerBottomLeft.score);
    printf("Bottom Right Score: %f\n", cornerBottomRight.score);
    printf("Corner Score: %f\n", cornerScore * 100);
    std::cout << "---" << std::endl;
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
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = guideFinderTopRight.x;
      y = guideFinderTopRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = guideFinderBottomLeft.x;
      y = guideFinderBottomLeft.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = guideFinderBottomRight.x;
      y = guideFinderBottomRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
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
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = detectionTopRight.x;
      y = detectionTopRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = detectionBottomLeft.x;
      y = detectionBottomLeft.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
      x = detectionBottomRight.x;
      y = detectionBottomRight.y;
      if (x - 2 + i >= 0 && x - 2 + i < frameWidth && y - 2 + j >= 0 &&
          y - 2 + j < frameHeight) {
        frameByteArrayOut[(y - 2 + j) * frameWidth + (x - 2 + i)] = 255;
      }
    }
  }

  // write image to a new pgm file with the same header as the input file
  std::ofstream outFile("assets/outputs/" + filename + "_out" +
                            std::to_string(mode) + ".pgm",
                        std::ios::binary);

  outFile << type << std::endl
          << frameWidth << " " << frameHeight << std::endl
          << intensity << std::endl;
  outFile.write((char *)frameByteArrayOut, frameWidth * frameHeight);
  outFile.close();

  // free memory
  delete[] frameByteArrayOut;

  return {cornerTopLeft, cornerTopRight, cornerBottomLeft, cornerBottomRight};
}

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

  std::vector<float> runtimes;
  std::vector<point_t> foundCorners;

  for (int i = 0; i < experiments; i++) {
    foundCorners = getCorners(runtimes, frameByteArray, width, height,
                              std::string(argv[1]), std::stoi(argv[2]));
  }

  // print averge runtime
  float sum = 0;
  for (int i = 0; i < experiments; i++) {
    sum += runtimes[i];
  }
  std::cout << "Average of " << experiments
            << " Runtimes: " << sum / runtimes.size() << " ms" << std::endl;

  delete[] frameByteArray;

  // return the number of found corners
  // note: the coordinates of the corners are in corner*** .x and .y members
  int numCorners = 0;
  for (int i = 0; i < 4; i++)
    if (foundCorners[i].score > 0)
      numCorners++;

  return numCorners;
}