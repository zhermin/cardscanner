
#ifndef __HOUGH_LINE_H_
#define __HOUGH_LINE_H_

#include <math.h>
#include <vector>

#ifndef PI
#define PI (3.1415926535897932384626433832795)
#endif

typedef enum _HOUGH_LINE_TYPE_CODE {
  HOUGH_LINE_STANDARD = 0,      // standard hough line
  HOUGH_LINE_PROBABILISTIC = 1, // probabilistic hough line

} HOUGH_LINE_TYPE_CODE;

typedef struct {
  int x;
  int y;
  int width;
  int height;
} boundingbox_t;

typedef struct {
  float startx;
  float starty;
  float endx;
  float endy;
} line_float_t;

typedef struct {
  int x;
  int y;
  float score;
} point_t;

/*
@function HoughLineDetector
@brief Custom C++ implementation of the Hough Line detection algorithm.
For HOUGH_LINE_STANDARD, the line points might fall outside of the image.
STANDARD EXAMPLE:
`HoughLineDetector(src, w, h, scaleX, scaleY, 70, 150, 1, PI / 180, 0, PI, 100,
                  HOUGH_LINE_STANDARD, bbox, lines)`
PROBABILISTIC EXAMPLE:
`HoughLineDetector(src, w, h, scaleX, scaleY, 70, 150, 1, PI / 180, 30, 10, 80,
                  HOUGH_LINE_PROBABILISTIC, bbox, lines)`

@param [in] src: image, single channel
@param [in] w: width of image
@param [in] h: height of image
@param [in] scaleX: downscale factor in X-axis
@param [in] scaleY: downscale factor in Y-axis
@param [in] sigma: Gaussian blur sigma value
@param [in] CannyLowThresh: lower threshold for hysteresis procedure in canny
@param [in] CannyHighThresh: higher threshold for hysteresis procedure in canny
@param [in] HoughRho: distance resolution of the accumulator in pixels
@param [in] HoughTheta: angle resolution of the accumulator in radians
@param [in] MinThetaLineLength:
STANDARD: for standard and multi-scale hough transform, minimum angle to check
for lines. PROBABILISTIC: minimum line length, line segments shorter than that
are rejected.
@param [in] MaxThetaGap:
STANDARD: for standard and multi-scale hough transform, maximum angle to check
for lines. PROBABILISTIC: maximum allowed gap between points on the same line to
link them.
@param [in] HoughThresh: accumulator threshold parameter, only lines that get
enough votes are returned ( >threshold )
@param [in] _type: HOUGH_LINE_STANDARD or HOUGH_LINE_PROBABILISTIC
@param [in] bbox: boundingbox or region to detect the lines in
@param [in/out] lines: detected lines are returned in this vector
@result [int] 0: success; 1: error
*/
int HoughLineDetector(unsigned char *src, int w, int h, float scaleX,
                      float scaleY, float sigma, float CannyLowThresh,
                      float CannyHighThresh, float HoughRho, float HoughTheta,
                      float MinThetaLinelength, float MaxThetaGap,
                      int HoughThresh, HOUGH_LINE_TYPE_CODE _type,
                      boundingbox_t bbox, std::vector<line_float_t> &lines);

#endif /* HOUGH_H */
