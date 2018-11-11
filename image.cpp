
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <random>
#include <iterator>
#include <math.h>
#include <time.h>
// #include <algorithm>
// #include <vector>
#include "image.h"

#define w 400

using namespace std;
using namespace cv;

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

inline float getSlope(const Point2f P1, const Point2f P2) {
	return (P2.y - P1.y) / (P2.x - P1.x);
}

inline float getIntercept(const float m, const Point2f P2) {
	return P2.y - m*P2.x;
}

// This function finds an intercept point of the normal from point P0
// to the estimated line model. P0 is taken from the testing set
inline Point2f getInterceptPoint(const float m, const float c,
	const Point2f P0) {
	Point2f p;
	p.x = (P0.x + m*P0.y - m*c) / (1 + m*m);
	p.y = (m*P0.x + m*m*P0.y - m*m*c)/(1 + m*m) + c;
	return p;
}

// RANSAC for 2D line fitting
Point2f ransac(const vector<Point2f> Points) {
	int iter = 100; // Reset this to 100 later
	float threshDist = 3.0; // Threshold of distances between points and line
	float ransacRatio = 0.50; // Threshold of number of inliers to assert model fits data well
	float numSamples = (float)Points.size();
	float bestM = 0; // Best slope
	float bestC = 0; // Best intercept
	float bestRatio = 0; // Best ratio
	float num;
	
	Point2f p1, p2;

	for (int i = 0; i < iter; i++) {
		p1 = *select_randomly(Points.begin(), Points.end());
		p2 = *select_randomly(Points.begin(), Points.end());
		while (p1 == p2)
			p2 = *select_randomly(Points.begin(), Points.end());
		float m = getSlope(p1, p2);
		float c = getIntercept(m, p2);
		cout << "[ INFO ] Iteration " << i << endl;

		vector<Point2f> inliers;
		num = 0;
		Point2f iP;
		float dist;
		for (Point2f candidate : Points) {
			if (candidate != p1 && candidate != p2) {
				iP = getInterceptPoint(m, c, candidate);
				dist = sqrt(pow(iP.x - candidate.x, 2) + 
					pow(iP.y - candidate.y, 2));
				if (dist < threshDist) {
					inliers.push_back(candidate);
					num++;
				}
			}
		}

		if (num/numSamples > bestRatio) {
			bestRatio = num/numSamples;
			bestM = m;
			bestC = c;
		}

		if (num > numSamples * ransacRatio) {
			cout << "Found good enough model" << endl;
			break;
		}
	}

	// Since we cannot return a tuple, we will encode the slope
	// and intercept as a Point2f(bestM, bestC). 
	Point2f res;
	res.x = bestM;
	res.y = bestC;
	return res;
}

float randFloat(float a, float b) {
    return ((b - a) * ((float)rand() / RAND_MAX)) + a;
}

inline float getY(const float m, const float c, const float x) {
	return x*m + c;
}

inline void createPoint(Mat img, Point center) {
  circle(img, center, 1, Scalar(0, 255, 0), FILLED, LINE_8);
}

inline void createLine( Mat img, Point start, Point end) {
  int thickness = 2;
  int lineType = LINE_8;
  line(img, start, end, Scalar(0, 0, 255), thickness, lineType);
}

void testRansac() {
	srand(time(0));
	int n = 200; // Number of points
	float inRatio = 0.6; // Ratio of inliers
	float m = randFloat(1, 3); // Random slope in range [1, 5]
	float c = randFloat(-5, 5); // Random intercept in range [-5, 5]
	int minX = (int)-n/2;
	int maxX = n - minX;
	int minY = minX;
	int maxY = maxX;
	int inlierDeviation = 10;
	int outlierDeviation = 200;

	cout << "m: " << m << ", c: " << c << endl;

	// Start the plot
	char plot[] = "Plot";
	Mat plotImg = Mat::zeros(w, w, CV_8UC3);


	vector<Point2f> data;
	float deviation;
	Point2f p;
	for (int i = minX; i < maxX; i++) {
		if (randFloat(0, 1) <= inRatio)
			deviation = inlierDeviation;
		else
			deviation = outlierDeviation;
		p.x = i + randFloat(-deviation, deviation);
		p.y = getY(m, c, p.x) + randFloat(-deviation, deviation);
		data.push_back(p);
		createPoint(plotImg, Point(p.x, p.y));
	}

	Point2f model = ransac(data);
	float modelM = model.x;
	float modelC = model.y;

	// Now plot the line model
	createLine(plotImg, Point(minX-50, getY(modelM, modelC, minX-50)),
		Point(maxX+50, getY(modelM, modelC, maxX+50)));
	imshow(plot, plotImg);
	waitKey();
}

// Correlation
double mean(const Image<float>& I,Point m,int n) {
	double s=0;
	for (int j=-n;j<=n;j++)
		for (int i=-n;i<=n;i++) 
			s+=I(m+Point(i,j));
	return s/(2*n+1)/(2*n+1);
}

double corr(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n) {
	double M1=mean(I1,m1,n);
	double M2=mean(I2,m2,n);
	double rho=0;
	for (int j=-n;j<=n;j++)
		for (int i=-n;i<=n;i++) {
			rho+=(I1(m1+Point(i,j))-M1)*(I2(m2+Point(i,j))-M2);
		}
		return rho;
}

double NCC(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n) {
	if (m1.x<n || m1.x>=I1.width()-n || m1.y<n || m1.y>=I1.height()-n) return -1;
	if (m2.x<n || m2.x>=I2.width()-n || m2.y<n || m2.y>=I2.height()-n) return -1;
	double c1=corr(I1,m1,I1,m1,n);
	if (c1==0) return -1;
	double c2=corr(I2,m2,I2,m2,n);
	if (c2==0) return -1;
	return corr(I1,m1,I2,m2,n)/sqrt(c1*c2);
}

