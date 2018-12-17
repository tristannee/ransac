#pragma once

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// Traits
template <typename T>
struct pixel_type
{
	static const int value = -1;
};
template <>
struct pixel_type<uchar>
{
	static const int value = CV_8U;
};
template <>
struct pixel_type<Vec3b>
{
	static const int value = CV_8UC3;
};
template <>
struct pixel_type<float>
{
	static const int value = CV_32F;
};

template <typename T> class Image : public Mat {
public:
	// Constructors
	Image() {}
	explicit Image(const Mat& A):Mat(A) {}
	Image(int w,int h):Mat(h,w,pixel_type<T>::value) {}
	// Accessors
	inline T operator()(int x,int y) const { return at<T>(y,x); }
	inline T& operator()(int x,int y) { return at<T>(y,x); }
	inline T operator()(const Point& p) const { return at<T>(p.y,p.x); }
	inline T& operator()(const Point& p) { return at<T>(p.y,p.x); }
	//
	inline int width() const { return cols; }
	inline int height() const { return rows; }
	// To display a floating type image
	Image<uchar> greyImage() const {
		double minVal, maxVal;
		minMaxLoc(*this,&minVal,&maxVal);
		Image<uchar> g;
		convertTo(g, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
		return g;
	}
};

Mat getPano();

void computeA(Mat &A, vector<Point2f> &obj, vector<Point2f> &scene, int a, int b, int c, int d);

Mat ransacGeneral(vector<Point2f> &obj, vector<Point2f> &scene);

Mat ransacGeneralAlternative(vector<Point2f> &obj, vector<Point2f> &scene);

Point2f ransac(const vector<Point2f> Points);

Mat testRansac();

// Correlation
double NCC(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n);
