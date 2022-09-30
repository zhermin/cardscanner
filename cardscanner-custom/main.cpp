#include "houghlines/houghlines.h"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

char type[10];
int height;
int width;
int intensity;

struct {
  bool debug = true;         // set to true to print debug info
  float resized_width = 300; // width of sized down image for faster processing
  float canny_lowerthreshold_ratio =
      0.2; // rejected if pixel gradient is below lower threshold (9)
  float canny_upperthreshold_ratio =
      0.6; // accepted if pixel gradient is above upper threshold (30)
  float houghline_threshold_ratio =
      0.1; // minimum intersections to detect a line (150)
  float houghline_minlinelength_ratio = 0.1; // minimum length of a line (60)
  float houghline_maxlinegap_ratio =
      0.1; // maximum gap between two points to form a line (2)
  float area_detection_ratio =
      0.10; // ratio of the detection area to the image area (30)
} PARAMS;

int main(int argc, char **argv) {
  // read image
  if (argc < 3) {
    cout << "Usage: $ " << argv[0]
         << " <input.pgm> <0/1 Standard/Probabilistic>" << endl;
    return 1;
  }

  ifstream infile(argv[1], ios::binary);
  if (!infile.is_open()) {
    cout << "Error opening file" << endl;
    return 1;
  }

  // read header
  infile >> type >> width >> height >> intensity;
  unsigned char *image = new unsigned char[width * height];
  infile.read((char *)image, width * height);
  infile.close();

  if (PARAMS.debug)
    cout << "Image size: " << width << "x" << height << endl;

  // detect lines
  vector<line_float_t> lines;
  boundingbox_t bbox;
  bbox.x = 0;
  bbox.y = 0;
  bbox.width = width;
  bbox.height = height;

  // calculate scale factor
  float scale = PARAMS.resized_width / (float)width;

  if (PARAMS.debug)
    cout << "Scale factor: " << scale << "x" << scale << endl;

  // use houghlines to detect lines
  if (stoi(argv[2]) == 0) {
    HoughLineDetector(image, width, height, scale, scale,
                      PARAMS.canny_lowerthreshold_ratio * PARAMS.resized_width,
                      PARAMS.canny_upperthreshold_ratio * PARAMS.resized_width,
                      1, M_PI / 180, 0, M_PI,
                      PARAMS.houghline_threshold_ratio * PARAMS.resized_width,
                      HOUGH_LINE_STANDARD, bbox, lines);
  } else {
    HoughLineDetector(
        image, width, height, scale, scale,
        PARAMS.canny_lowerthreshold_ratio * PARAMS.resized_width,
        PARAMS.canny_upperthreshold_ratio * PARAMS.resized_width, 1, M_PI / 180,
        PARAMS.houghline_minlinelength_ratio * PARAMS.resized_width,
        PARAMS.houghline_maxlinegap_ratio * PARAMS.resized_width,
        PARAMS.houghline_threshold_ratio * PARAMS.resized_width,
        HOUGH_LINE_PROBABILISTIC, bbox, lines);
  }

  // only keep the lines in the detection regions (top, bottom, left, right)
  vector<line_float_t> lines_top;
  vector<line_float_t> lines_bottom;
  vector<line_float_t> lines_left;
  vector<line_float_t> lines_right;
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
  vector<line_float_t> foundlines;
  foundlines.insert(foundlines.end(), lines_top.begin(), lines_top.end());
  foundlines.insert(foundlines.end(), lines_bottom.begin(), lines_bottom.end());
  foundlines.insert(foundlines.end(), lines_left.begin(), lines_left.end());
  foundlines.insert(foundlines.end(), lines_right.begin(), lines_right.end());

  // print lines
  if (PARAMS.debug) {
    cout << "Found " << lines.size() << " lines (All)" << endl;
    cout << "Found " << foundlines.size() << " lines (Sides)" << endl;
  }

  // print all lines
  if (PARAMS.debug) {
    for (int i = 0; i < foundlines.size(); i++) {
      cout << foundlines[i].startx << " " << foundlines[i].starty << " "
           << foundlines[i].endx << " " << foundlines[i].endy << endl;
    }
  }

  // check for corners
  // top left corner: if there is a line with startx and starty within detection
  // area ratio multiplied by the image width and height, and so on for the rest
  point_t corner_top_left, corner_top_right, corner_bottom_left,
      corner_bottom_right;
  bool found_top_left = false;
  bool found_top_right = false;
  bool found_bottom_left = false;
  bool found_bottom_right = false;
  for (int i = 0; i < foundlines.size(); i++) {
    line_float_t line = foundlines[i];
    int minx = min(line.startx, line.endx);
    int miny = min(line.starty, line.endy);
    int maxx = max(line.startx, line.endx);
    int maxy = max(line.starty, line.endy);
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

  // print corners
  if (PARAMS.debug) {
    cout << "Found corners: " << endl;
    string print_top_left = found_top_left
                                ? "Top left: " + to_string(found_top_left) +
                                      " (" + to_string(corner_top_left.x) +
                                      ", " + to_string(corner_top_left.y) + ")"
                                : "Top left: " + to_string(found_top_left);
    string print_top_right =
        found_top_right ? "Top right: " + to_string(found_top_right) + " (" +
                              to_string(corner_top_right.x) + ", " +
                              to_string(corner_top_right.y) + ")"
                        : "Top right: " + to_string(found_top_right);
    string print_bottom_left =
        found_bottom_left ? "Bottom left: " + to_string(found_bottom_left) +
                                " (" + to_string(corner_bottom_left.x) + ", " +
                                to_string(corner_bottom_left.y) + ")"
                          : "Bottom left: " + to_string(found_bottom_left);
    string print_bottom_right =
        found_bottom_right ? "Bottom right: " + to_string(found_bottom_right) +
                                 " (" + to_string(corner_bottom_right.x) +
                                 ", " + to_string(corner_bottom_right.y) + ")"
                           : "Bottom right: " + to_string(found_bottom_right);
    cout << print_top_left << endl;
    cout << print_top_right << endl;
    cout << print_bottom_left << endl;
    cout << print_bottom_right << endl;

    // if 3 or more corners are found, a card is detected
    int found_corners = int(found_top_left) + int(found_top_right) +
                        int(found_bottom_left) + int(found_bottom_right);
    if (found_corners >= 3) {
      cout << "Card detected!" << endl;
    } else {
      cout << "Card NOT detected!" << endl;
    }
  }

  // draw the found lines on a blank image
  unsigned char *image_out = new unsigned char[width * height];
  for (int i = 0; i < width * height; i++) {
    image_out[i] = 0;
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
      image_out[y1 * width + x1] = 255;
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

  // write image to a new pgm file with the same header as the input file
  string outfilename;

  // remove the extension from the input filename
  size_t lastindex = string(argv[1]).find_last_of(".");
  string filename = string(argv[1]).substr(0, lastindex);

  if (stoi(argv[2]) == 0) {
    outfilename = filename + "_out" + string(argv[2]) + ".pgm";
  } else {
    outfilename = filename + "_out" + string(argv[2]) + ".pgm";
  }

  if (PARAMS.debug)
    cout << "Writing output to " << outfilename << endl;

  ofstream outfile(outfilename, ios::binary);
  outfile << type << endl
          << width << " " << height << endl
          << intensity << endl;
  outfile.write((char *)image_out, width * height);
  outfile.close();

  // free memory
  delete[] image;
  delete[] image_out;

  return 0;
}