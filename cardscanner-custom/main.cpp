#include "houghlines/houghlines.h"
#include <fstream>

char type[10];
int width;
int height;
int intensity;

struct {
  float resized_width =
      300; // new width of sized down image for faster processing
  float area_detection_ratio =
      0.10; // ratio of the detection area to the image area (30)
  float canny_lowerthreshold_ratio =
      0.2; // rejected if pixel gradient is below lower threshold (60)
  float canny_upperthreshold_ratio =
      0.6; // accepted if pixel gradient is above upper threshold (180)
  float houghline_threshold_ratio =
      0.1; // minimum intersections to detect a line (30)
  float houghline_minlinelength_ratio =
      0.1; // minimum length of a line to detect (30)
  float houghline_maxlinegap_ratio =
      0.1; // maximum gap between two potential lines to join into 1 line (3)
} PARAMS;

int main(int argc, char **argv) {
  // USAGE
  // $ ./main.o <pgm image (in ./assets folder)> <0/1 Standard/Probabilistic>
  // EXAMPLE
  // $ ./main.o card5 1; echo $?
  // >> -1 (error) or 0-4 (number of found corners)

  if (argc < 3)
    return -1;

  std::ifstream infile("assets/" + std::string(argv[1]) + ".pgm",
                       std::ios::binary);
  if (!infile.is_open())
    return -1;

  // read header (.pgm type and intensity header info are unused)
  infile >> type >> width >> height >> intensity;
  unsigned char *frameByteArray = new unsigned char[width * height];
  infile.read((char *)frameByteArray, width * height);
  infile.close();

  // initialize lines vector and bounding box
  std::vector<line_float_t> lines;
  boundingbox_t bbox;
  bbox.x = 0;
  bbox.y = 0;
  bbox.width = width;
  bbox.height = height;

  // calculate scale factor
  float scale = PARAMS.resized_width / (float)width;

  // use houghlines to detect lines
  if (std::stoi(argv[2]) == 0) {
    HoughLineDetector(frameByteArray, width, height, scale, scale,
                      PARAMS.canny_lowerthreshold_ratio * PARAMS.resized_width,
                      PARAMS.canny_upperthreshold_ratio * PARAMS.resized_width,
                      1, PI / 180, 0, PI,
                      PARAMS.houghline_threshold_ratio * PARAMS.resized_width,
                      HOUGH_LINE_STANDARD, bbox, lines);
  } else {
    HoughLineDetector(
        frameByteArray, width, height, scale, scale,
        PARAMS.canny_lowerthreshold_ratio * PARAMS.resized_width,
        PARAMS.canny_upperthreshold_ratio * PARAMS.resized_width, 1, PI / 180,
        PARAMS.houghline_minlinelength_ratio * PARAMS.resized_width,
        PARAMS.houghline_maxlinegap_ratio * PARAMS.resized_width,
        PARAMS.houghline_threshold_ratio * PARAMS.resized_width,
        HOUGH_LINE_PROBABILISTIC, bbox, lines);
  }

  // only keep the lines in the detection regions (top, bottom, left, right)
  std::vector<line_float_t> lines_top;
  std::vector<line_float_t> lines_bottom;
  std::vector<line_float_t> lines_left;
  std::vector<line_float_t> lines_right;
  for (int i = 0; i < lines.size(); i++) {
    line_float_t line = lines[i];
    if (line.startx == 1 && line.endx == 1)
      continue;
    if (line.startx < width * PARAMS.area_detection_ratio &&
        line.endx < width * PARAMS.area_detection_ratio) {
      lines_left.push_back(line);
    } else if (line.startx > width * (1 - PARAMS.area_detection_ratio) &&
               line.endx > width * (1 - PARAMS.area_detection_ratio)) {
      lines_right.push_back(line);
    } else if (line.starty < height * PARAMS.area_detection_ratio &&
               line.endy < height * PARAMS.area_detection_ratio) {
      lines_top.push_back(line);
    } else if (line.starty > height * (1 - PARAMS.area_detection_ratio) &&
               line.endy > height * (1 - PARAMS.area_detection_ratio)) {
      lines_bottom.push_back(line);
    }
  }

  // combine the region lines
  std::vector<line_float_t> foundlines;
  foundlines.insert(foundlines.end(), lines_top.begin(), lines_top.end());
  foundlines.insert(foundlines.end(), lines_bottom.begin(), lines_bottom.end());
  foundlines.insert(foundlines.end(), lines_left.begin(), lines_left.end());
  foundlines.insert(foundlines.end(), lines_right.begin(), lines_right.end());

  // check in all 4 corners if there is a line with x and y coordinates
  // within detection area ratio multiplied by the image width and height
  point_t corner_top_left, corner_top_right, corner_bottom_left,
      corner_bottom_right;
  bool found_top_left = false;
  bool found_top_right = false;
  bool found_bottom_left = false;
  bool found_bottom_right = false;
  for (int i = 0; i < foundlines.size(); i++) {
    line_float_t line = foundlines[i];
    int minx = std::min(line.startx, line.endx);
    int miny = std::min(line.starty, line.endy);
    int maxx = std::max(line.startx, line.endx);
    int maxy = std::max(line.starty, line.endy);
    if (minx < width * PARAMS.area_detection_ratio &&
        miny < height * PARAMS.area_detection_ratio) {
      corner_top_left.x = minx;
      corner_top_left.y = miny;
      found_top_left = true;
    } else if (maxx > width * (1 - PARAMS.area_detection_ratio) &&
               miny < height * PARAMS.area_detection_ratio) {
      corner_top_right.x = maxx;
      corner_top_right.y = miny;
      found_top_right = true;
    } else if (minx < width * PARAMS.area_detection_ratio &&
               maxy > height * (1 - PARAMS.area_detection_ratio)) {
      corner_bottom_left.x = minx;
      corner_bottom_left.y = maxy;
      found_bottom_left = true;
    } else if (maxx > width * (1 - PARAMS.area_detection_ratio) &&
               maxy > height * (1 - PARAMS.area_detection_ratio)) {
      corner_bottom_right.x = maxx;
      corner_bottom_right.y = maxy;
      found_bottom_right = true;
    }
  }

  // draw the found lines on a blank image
  unsigned char *frameByteArrayOut = new unsigned char[width * height];
  for (int i = 0; i < width * height; i++) {
    frameByteArrayOut[i] = 0;
  }
  for (int i = 0; i < foundlines.size(); i++) {
    int x1 = foundlines[i].startx;
    int y1 = foundlines[i].starty;
    int x2 = foundlines[i].endx;
    int y2 = foundlines[i].endy;
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
      if (found_top_left) {
        frameByteArrayOut[(corner_top_left.y - 2 + i) * width +
                          (corner_top_left.x - 2 + j)] = 255;
      }
      if (found_top_right) {
        frameByteArrayOut[(corner_top_right.y - 2 + i) * width +
                          (corner_top_right.x - 2 + j)] = 255;
      }
      if (found_bottom_left) {
        frameByteArrayOut[(corner_bottom_left.y - 2 + i) * width +
                          (corner_bottom_left.x - 2 + j)] = 255;
      }
      if (found_bottom_right) {
        frameByteArrayOut[(corner_bottom_right.y - 2 + i) * width +
                          (corner_bottom_right.x - 2 + j)] = 255;
      }
    }
  }

  // write image to a new pgm file with the same header as the input file
  std::ofstream outfile("assets/outputs/" + std::string(argv[1]) + "_out" +
                            std::string(argv[2]) + ".pgm",
                        std::ios::binary);

  outfile << type << std::endl
          << width << " " << height << std::endl
          << intensity << std::endl;
  outfile.write((char *)frameByteArrayOut, width * height);
  outfile.close();

  // free memory
  delete[] frameByteArray;
  delete[] frameByteArrayOut;

  // if 3 or more corners are found, a card is detected
  int found_corners = int(found_top_left) + int(found_top_right) +
                      int(found_bottom_left) + int(found_bottom_right);

  // return the number of found corners
  // note: the coordinates of the corners are in corner_*** .x and .y members
  return found_corners;
}