#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

// Note: Ratio params are based off `max_size`, actual values are in brackets
struct {
  bool debug = false;   // set to true to show found lines and corners
  float max_size = 300; // max width of scaled down image
  float frame_scaling_factor =
      0.6; // ratio of unmasked area to the entire frame (180)
  float mask_aspect_ratio_width = 86;  // CR80 standard width is 86mm
  float mask_aspect_ratio_height = 54; // CR80 standard height is 54mm
  int gaussian_blur_radius = 5;        // higher radius = more blur
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

Mat process_image(Mat &img) {
  // Resize image
  Mat img_resized;
  resize(img, img_resized,
         Size(PARAMS.max_size, cvRound(PARAMS.max_size / img.cols * img.rows)));

  // Convert to grayscale
  Mat img_gray;
  cvtColor(img_resized, img_gray, COLOR_BGR2GRAY);

  // Apply Gaussian blur
  Mat img_blur;
  GaussianBlur(img_gray, img_blur,
               Size(PARAMS.gaussian_blur_radius, PARAMS.gaussian_blur_radius),
               0);

  // Apply Canny edge detection
  Mat img_canny;
  Canny(img_blur, img_canny,
        PARAMS.canny_lowerthreshold_ratio * PARAMS.max_size,
        PARAMS.canny_upperthreshold_ratio * PARAMS.max_size);

  // Dilate image
  Mat img_dilate;
  Mat kernel = getStructuringElement(
      MORPH_RECT, Size(PARAMS.dilate_kernel_size, PARAMS.dilate_kernel_size));
  dilate(img_canny, img_dilate, kernel);

  return img_dilate;
}

vector<Vec4i> get_houghlines(Mat &region) {
  vector<Vec4i> lines;
  HoughLinesP(region, lines, 1, CV_PI / 180,
              PARAMS.houghline_threshold_ratio * PARAMS.max_size,
              PARAMS.houghline_minlinelength_ratio * PARAMS.max_size,
              PARAMS.houghline_maxlinegap_ratio * PARAMS.max_size);
  return lines;
}

int find_corners(Mat &img_processed) {
  // Find lines in cropped regions
  int imgH = img_processed.rows, imgW = img_processed.cols;
  int detectionH = cvRound(imgH * PARAMS.area_detection_ratio);
  int detectionW = cvRound(imgW * PARAMS.area_detection_ratio);

  // Left rectangle region
  Rect rect_left(0, 0, detectionW, imgH);
  Mat region_left = img_processed(rect_left);
  // Right rectangle region
  Rect rect_right(imgW - detectionW, 0, detectionW, imgH);
  Mat region_right = img_processed(rect_right);
  // Top rectangle region
  Rect rect_top(0, 0, imgW, detectionH);
  Mat region_top = img_processed(rect_top);
  // Bottom rectangle region
  Rect rect_bottom(0, imgH - detectionH, imgW, detectionH);
  Mat region_bottom = img_processed(rect_bottom);

  // Apply Probabilistic Hough Transform to find lines
  vector<Vec4i> lines_left = get_houghlines(region_left);
  vector<Vec4i> lines_right = get_houghlines(region_right);
  vector<Vec4i> lines_top = get_houghlines(region_top);
  vector<Vec4i> lines_bottom = get_houghlines(region_bottom);

  // Draw found lines onto a new blank image to find corners later
  Mat highlighted_lines = Mat::zeros(img_processed.size(), CV_8UC3);
  for (size_t i = 0; i < lines_left.size(); i++) {
    Vec4i l = lines_left[i];
    line(highlighted_lines, Point(l[0], l[1]), Point(l[2], l[3]),
         Scalar(0, 0, 255), 1, LINE_AA);
  }
  for (size_t i = 0; i < lines_right.size(); i++) {
    Vec4i l = lines_right[i];
    line(highlighted_lines, Point(l[0] + imgW - detectionW, l[1]),
         Point(l[2] + imgW - detectionW, l[3]), Scalar(0, 0, 255), 1, LINE_AA);
  }
  for (size_t i = 0; i < lines_top.size(); i++) {
    Vec4i l = lines_top[i];
    line(highlighted_lines, Point(l[0], l[1]), Point(l[2], l[3]),
         Scalar(0, 0, 255), 1, LINE_AA);
  }
  for (size_t i = 0; i < lines_bottom.size(); i++) {
    Vec4i l = lines_bottom[i];
    line(highlighted_lines, Point(l[0], l[1] + imgH - detectionH),
         Point(l[2], l[3] + imgH - detectionH), Scalar(0, 0, 255), 1, LINE_AA);
  }

  // Further crop the 4 sides into 4 corners
  Mat corner_frame;
  cvtColor(highlighted_lines, corner_frame, COLOR_BGR2GRAY);

  // Top left corner
  Rect rect_topleft = Rect(0, 0, detectionW, detectionH);
  Mat region_topleft = corner_frame(rect_topleft);
  // Top right corner
  Rect rect_topright = Rect(imgW - detectionW, 0, detectionW, detectionH);
  Mat region_topright = corner_frame(rect_topright);
  // Bottom left corner
  Rect rect_bottomleft = Rect(0, imgH - detectionH, detectionW, detectionH);
  Mat region_bottomleft = corner_frame(rect_bottomleft);
  // Bottom right corner
  Rect rect_bottomright =
      Rect(imgW - detectionW, imgH - detectionH, detectionW, detectionH);
  Mat region_bottomright = corner_frame(rect_bottomright);

  // Apply Shi-Tomasi corner detection to find corners
  vector<Point2f> corner_topleft;
  goodFeaturesToTrack(region_topleft, corner_topleft, 1,
                      PARAMS.corner_quality_ratio, 20);
  vector<Point2f> corner_topright;
  goodFeaturesToTrack(region_topright, corner_topright, 1,
                      PARAMS.corner_quality_ratio, 20);
  vector<Point2f> corner_bottomleft;
  goodFeaturesToTrack(region_bottomleft, corner_bottomleft, 1,
                      PARAMS.corner_quality_ratio, 20);
  vector<Point2f> corner_bottomright;
  goodFeaturesToTrack(region_bottomright, corner_bottomright, 1,
                      PARAMS.corner_quality_ratio, 20);

  // Draw found corners onto the highlighted lines image
  if (PARAMS.debug) {
    for (size_t i = 0; i < corner_topleft.size(); i++) {
      Point2f p = corner_topleft[i];
      circle(highlighted_lines, Point(p.x, p.y), 5, Scalar(0, 255, 0), 2);
    }
    for (size_t i = 0; i < corner_topright.size(); i++) {
      Point2f p = corner_topright[i];
      circle(highlighted_lines, Point(p.x + imgW - detectionW, p.y), 5,
             Scalar(0, 255, 0), 2);
    }
    for (size_t i = 0; i < corner_bottomleft.size(); i++) {
      Point2f p = corner_bottomleft[i];
      circle(highlighted_lines, Point(p.x, p.y + imgH - detectionH), 5,
             Scalar(0, 255, 0), 2);
    }
    for (size_t i = 0; i < corner_bottomright.size(); i++) {
      Point2f p = corner_bottomright[i];
      circle(highlighted_lines,
             Point(p.x + imgW - detectionW, p.y + imgH - detectionH), 5,
             Scalar(0, 255, 0), 2);
    }

    imshow("Highlighted Lines", highlighted_lines);
  };

  return (int)(corner_topleft.size() + corner_topright.size() +
               corner_bottomleft.size() + corner_bottomright.size());
}

int main() {
  string app_name = "Card Scanner";

  // Load camera or video using OpenCV
  VideoCapture cap(0);
  Mat img;

  // Calculate number of frames to wait for valid frames
  int valid_frames = 0;

  cout << "Initialized " << app_name << "! Press ESC to quit" << endl;

  while (true) {
    // Read frame from camera or video
    cap.read(img);
    if (img.empty()) {
      cout << "Could not read frame, exiting..." << endl;
      break;
    }

    float camH = img.rows, camW = img.cols;

    // Crop image to the card aspect ratio
    // Note: No mask applied in this implementation
    Mat cropped;
    float mask_aspect_ratio =
        PARAMS.mask_aspect_ratio_height / PARAMS.mask_aspect_ratio_width;
    if (camH / camW > mask_aspect_ratio) {
      // Crop top and bottom if image height is longer
      int cropH = (camH - camW * mask_aspect_ratio) / 2;
      Rect crop_rect = Rect(0, cropH, camW, camH - 2 * cropH);
      cropped = img(crop_rect);
    } else if (camH / camW < mask_aspect_ratio) {
      // Crop left and right if image width is longer
      int cropW = (camW - camH / mask_aspect_ratio) / 2;
      Rect crop_rect = Rect(cropW, 0, camW - 2 * cropW, camH);
      cropped = img(crop_rect);
    }

    // Process image
    Mat img_processed = process_image(cropped);
    if (PARAMS.debug) {
      imshow("Processed Image", img_processed);
    }

    // Find corners using Hough Lines Transform and Shi-Tomasi corner detection
    int corner_count = find_corners(img_processed);

    // Save a screenshot if at least 3 corners are found (card detected)
    // after Y milliseconds (calculated from 30 FPS in this implementation)
    if (corner_count >= 3) {
      valid_frames++;
      if (valid_frames >= PARAMS.wait_frames) {
        valid_frames = 0;
        imshow("Auto Captured Card", cropped);
        waitKey(0);
        destroyWindow("Auto Captured Card");
        //        imwrite("card.png", img);
      }
    } else {
      valid_frames = 0;
    }

    // Show cropped webcame or video to the user
    imshow(app_name, cropped);

    // Press ESC on keyboard to exit
    if (waitKey(1) == 27) {
      cout << "Shutting down " << app_name << endl;
      cap.release();
      break;
    };
  }

  return 0;
}
