#include <fstream>
#include <iostream>

using namespace std;

char type[10];
int height;
int width;
int intensity;

static struct {
  bool debug = false;        // set to true to show found lines and corners
  float resized_width = 300; // width of sized down image for faster processing
  float mask_aspect_ratio_width = 86;  // CR80 standard width is 86mm
  float mask_aspect_ratio_height = 54; // CR80 standard height is 54mm
  int gaussian_sigma = 1;              // higher sigma value = more blur
  float canny_lowerthreshold_ratio =
      0.03; // rejected if pixel gradient is below lower threshold (9)
  float canny_upperthreshold_ratio =
      0.10; // accepted if pixel gradient is above upper threshold (30)
  int dilate_kernel_size = 3; // larger kernel = thicker lines
  float houghline_threshold_ratio =
      0.5; // minimum intersections to detect a line (150)
  float houghline_minlinelength_ratio = 0.2; // minimum length of a line (60)
  float houghline_maxlinegap_ratio =
      0.005; // maximum gap between two points to form a line (2)
  float area_detection_ratio =
      0.10; // ratio of the detection area to the image area (30)
  float corner_quality_ratio = 0.99; // higher value = stricter corner detection
  int wait_frames = 0; // number of consecutive valid frames to wait
} PARAMS;

int main() {

  // Open file in binary mode
  ifstream infile("../card.pgm", ios::binary);
  infile >> ::type >> ::width >> ::height >> ::intensity;
  cout << "Type: " << ::type << endl;
  cout << "Width: " << ::width << endl;
  cout << "Height: " << ::height << endl;
  cout << "Intensity: " << ::intensity << endl;

  // Copy .pgm header information from input file to output file
  ofstream outfile("output.pgm", ios::binary);
  outfile << ::type << endl
          << ::width << " " << ::height << endl
          << ::intensity << endl;

  // ofstream myfile;
  // myfile.open ("example.txt");
  // myfile << "Writing this to a file.\n";
  // myfile.close();
  return 0;
}