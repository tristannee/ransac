#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/stitching.hpp"
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

//template<class BidiIter >
//BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random) {
//    size_t left = std::distance(begin, end);
//    while (num_random--) {
//        BidiIter r = begin;
//        std::advance(r, rand()%left);
//        std::swap(begin, r);
//        ++begin;
//        --left;
//    }
//    return begin;
//}

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
        // cout << "[ INFO ] Iteration " << i << endl;
        
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
            // cout << "Found good enough model" << endl;
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

//void computeH(Mat &H, Point2f a, Point2f ap,Point2f b, Point2f bp,Point2f c, Point2f cp, Point2f d, Point2f dp){
//    H = (Mat_<double>(9,9) << 0, 0, 0, -a.x, -a.y, -1, ap.y*a.x, ap.y*a.y, ap.y,
//                              a.x, a.y, 1, 0, 0, 0, -a.x*ap.x, -ap.x*a.y, -ap.x,
//                              0, 0, 0, -b.x, -a.y, -1, bp.y*a.x, bp.y*a.y, bp.y,
//                              b.x, b.y, 1, 0, 0, 0, -b.x*bp.x, -bp.x*b.y, -bp.x,
//                              0, 0, 0, -c.x, -c.y, -1, cp.y*c.x, cp.y*c.y, cp.y,
//                              c.x, c.y, 1, 0, 0, 0, -c.x*cp.x, -cp.x*c.y, -cp.x,
//                              0, 0, 0, -d.x, -d.y, -1, dp.y*d.x, dp.y*a.y, dp.y,
//                              d.x, d.y, 1, 0, 0, 0, -d.x*ap.x, -dp.x*d.y, -dp.x);
//}

void computeA(Mat &A, vector<Point2f> &obj, vector<Point2f> &scene, int a, int b, int c, int d){
    
    
    
    A = (Mat_<float>(8,9) << 0, 0, 0, -obj[a].x, -obj[a].y, -1, scene[a].y*obj[a].x, scene[a].y*obj[a].y, scene[a].y,
         obj[a].x, obj[a].y, 1, 0, 0, 0, -obj[a].x*scene[a].x, -scene[a].x*obj[a].y, -scene[a].x,
         0, 0, 0, -obj[b].x, -obj[b].y, -1, scene[b].y*obj[b].x, scene[b].y*obj[b].y, scene[b].y,
         obj[b].x, obj[b].y, 1, 0, 0, 0, -obj[b].x*scene[b].x, -scene[b].x*obj[b].y, -scene[b].x,
         0, 0, 0, -obj[c].x, -obj[c].y, -1, scene[c].y*obj[c].x, scene[c].y*obj[c].y, scene[b].y,
         obj[c].x, obj[c].y, 1, 0, 0, 0, -obj[c].x*scene[c].x, -scene[c].x*obj[c].y, -scene[c].x,
         0, 0, 0, -obj[d].x, -obj[d].y, -1, scene[d].y*obj[d].x, scene[d].y*obj[d].y, scene[d].y,
         obj[d].x, obj[d].y, 1, 0, 0, 0, -obj[d].x*scene[d].x, -scene[d].x*obj[d].y, -scene[d].x);
}

void computeAAlternative(Mat &A, vector<Point2f> &obj, vector<Point2f> &scene, int a, int b, int c, int d){
    
    
    
    A = (Mat_<float>(9, 9) << -obj[a].x, -obj[a].y, -1, 0, 0, 0, obj[a].x*scene[a].x, obj[a].y*scene[a].x, scene[a].x,
         0, 0, 0, -obj[a].x, -obj[a].y, -1, obj[a].x*scene[a].y, obj[a].y*scene[a].y, scene[a].y,
         -obj[b].x, -obj[b].y, -1, 0, 0, 0, obj[b].x*scene[b].x, obj[b].y*scene[b].x, scene[b].x,
         0, 0, 0, -obj[b].x, -obj[b].y, -1, obj[b].x*scene[b].y, obj[b].y*scene[b].y, scene[b].y,
         -obj[c].x, -obj[c].y, -1, 0, 0, 0, obj[c].x*scene[c].x, obj[c].y*scene[c].x, scene[c].x,
         0, 0, 0, -obj[c].x, -obj[c].y, -1, obj[c].x*scene[c].y, obj[c].y*scene[c].y, scene[c].y,
         -obj[d].x, -obj[d].y, -1, 0, 0, 0, obj[d].x*scene[d].x, obj[d].y*scene[d].x, scene[d].x,
         0, 0, 0, -obj[d].x, -obj[d].y, -1, obj[d].x*scene[d].y, obj[d].y*scene[d].y, scene[d].y,
         0, 0, 0, 0, 0, 0, 0, 0, 1);
}



void computeAAAlternative(Mat &A, vector<Point2f> &obj, vector<Point2f> &scene, int a, int b, int c, int d){
    
    
    
    A = (Mat_<float>(9,9) << 0, 0, 0, -obj[a].x, -obj[a].y, -1, scene[a].y*obj[a].x, scene[a].y*obj[a].y, scene[a].y,
         obj[a].x, obj[a].y, 1, 0, 0, 0, -obj[a].x*scene[a].x, -scene[a].x*obj[a].y, -scene[a].x,
         0, 0, 0, -obj[b].x, -obj[b].y, -1, scene[b].y*obj[b].x, scene[b].y*obj[b].y, scene[b].y,
         obj[b].x, obj[b].y, 1, 0, 0, 0, -obj[b].x*scene[b].x, -scene[b].x*obj[b].y, -scene[b].x,
         0, 0, 0, -obj[c].x, -obj[c].y, -1, scene[c].y*obj[c].x, scene[c].y*obj[c].y, scene[b].y,
         obj[c].x, obj[c].y, 1, 0, 0, 0, -obj[c].x*scene[c].x, -scene[c].x*obj[c].y, -scene[c].x,
         0, 0, 0, -obj[d].x, -obj[d].y, -1, scene[d].y*obj[d].x, scene[d].y*obj[d].y, scene[d].y,
         obj[d].x, obj[d].y, 1, 0, 0, 0, -obj[d].x*scene[d].x, -scene[d].x*obj[d].y, -scene[d].x,
         0, 0, 0, 0, 0, 0, 0, 0, 1);
}


Mat ransacGeneralAlternative(vector<Point2f> &obj, vector<Point2f> &scene) {
    int iter = 1000; // Reset this to 100 later
    float threshDist = 1.0; // Threshold of distances between points and line
    float ransacRatio = 0.80; // Threshold of number of inliers to assert model fits data well
    float numSamples = (float)obj.size();
    float bestRatio = 0; // Best ratio
    float num;
    
    Point2f p1, p2;
    
    //solve(InputArray src1, InputArray src2, OutputArray dst, int flags=DECOMP_LU);
    
    Mat_<float> A, H, Z, bestH, s, u, vt, v;
    Z = Mat::zeros(1,9, CV_32F);
    Z.at<float>(8)=1;
    // cout << Z << endl;
    
    
    int sz = obj.size();
    
    int a,b,c,d;
    vector<int> inliers;
    
    vector<float> objPoint;
    vector<float> scnPoint;
    Z = Z.t();
    
    for (int i = 0; i < iter; i++) {
        //        random_unique(Points.begin(), Points.end(), 4);
        a = rand() % sz;
        b = rand() % sz;
        while (b == a){
            b = rand() % sz;
        }
        c = rand() % sz;
        while (c == a || c == b){
            c = rand() % sz;
        }
        d = rand() % sz;
        while (d == a || d == b || d == c){
            d = rand() % sz;
        }
        
        computeAAlternative(A, obj, scene, a, b, c, d); // AH = 0
        //        cout << A << endl;
        //SVD::compute(A, s, u, vt);
//        cout << vt.size()  << endl;
//
//        cout << s << endl;
//        cout << vt << endl;
//        vt.row(7).copyTo(H);
//        cout << H << endl;
        // cout <<a<<endl;
        
        solve(A, Z, H, DECOMP_SVD);
        // cout << H.size()  << endl;
        if(H.size().width!=1) continue;
        H = H.reshape(0, 3);
        // cout << H << endl;
        //        cout << H.size()  << endl;
        // cout << "[ INFO ] Iteration " << i << endl;
        
        
        num = 0;
        Point2f iP;
        float dist;
        for (int i = 0; i<obj.size(); i++) {
            if(i == a || i == b || i == c || i == d){
                continue;
            }
            objPoint.clear();
            scnPoint.clear();
            
            //                iP = getInterceptPoint(m, c, candidate);
            objPoint.push_back(obj[i].x);
            objPoint.push_back(obj[i].y);
            objPoint.push_back(1);
            
            scnPoint.push_back(scene[i].x);
            scnPoint.push_back(scene[i].y);
            scnPoint.push_back(1);
            
            
            double dist = norm(Mat_<float>(scnPoint), H * Mat_<float>(objPoint));
            // cout << "dist " << dist << endl;
            
            //            dist = sqrt(pow(iP.x - candidate.x, 2) +
            //                        pow(iP.y - candidate.y, 2));
            if (dist < threshDist) {
                inliers.push_back(i);
                num++;
            }
            
        }
        
        // cout << "num/numSamples " << num/numSamples << endl;
        if (num/numSamples > bestRatio) {
            bestRatio = num/numSamples;
            bestH = H;
        }
        
        if (num > numSamples * ransacRatio) {
            // cout << "Found good enough model" << endl;
            break;
        }
    }
    
    // Since we cannot return a tuple, we will encode the slope
    // and intercept as a Point2f(bestM, bestC).
    
    // cout << bestH << endl;
    return bestH;
}


Mat ransacGeneral(vector<Point2f> &obj, vector<Point2f> &scene) {
    int iter = 1000; // Reset this to 100 later
    float threshDist = 100; // Threshold of distances between points and line
    float ransacRatio = 0.50; // Threshold of number of inliers to assert model fits data well
    float numSamples = (float)obj.size();
    float bestRatio = 0; // Best ratio
    float num;
    
    Point2f p1, p2;
    
    //solve(InputArray src1, InputArray src2, OutputArray dst, int flags=DECOMP_LU);
    
    Mat_<float> A, H, Z, bestH, s, u, vt, v;
    Z = Mat::zeros(1,9, CV_32F);
    
    int sz = obj.size();
    
    int a,b,c,d;
    vector<int> inliers;
    
    vector<float> objPoint;
    vector<float> scnPoint;
    
    for (int i = 0; i < iter; i++) {
//        random_unique(Points.begin(), Points.end(), 4);
        a = rand() % sz;
        b = rand() % sz;
        while (b == a){
            b = rand() % sz;
        }
        c = rand() % sz;
        while (c == a || c == b){
            c = rand() % sz;
        }
        d = rand() % sz;
        while (d == a || d == b || d == c){
            d = rand() % sz;
        }
        computeA(A, obj, scene, a, b, c, d); // AH = 0
//        cout << A << endl;
        SVD::compute(A, s, u, vt);
        // cout << vt.size()  << endl;
        
        // cout << s << endl;
        // cout << vt << endl;
        vt.row(7).copyTo(H);
        // cout << H << endl;
        
//        solve(A, Z, H, DECOMP_SVD);
//        cout << H.size()  << endl;
        H = H.reshape(0, 3);
        // cout << H << endl;
//        cout << H.size()  << endl;
        // cout << "[ INFO ] Iteration " << i << endl;
        
        
        num = 0;
        Point2f iP;
        float dist;
        for (int i = 0; i<obj.size(); i++) {
            if(i == a || i == b || i == c || i == d){
                continue;
            }
            objPoint.clear();
            scnPoint.clear();
            
//                iP = getInterceptPoint(m, c, candidate);
            objPoint.push_back(obj[i].x);
            objPoint.push_back(obj[i].y);
            objPoint.push_back(1);
            
            scnPoint.push_back(scene[i].x);
            scnPoint.push_back(scene[i].y);
            scnPoint.push_back(1);
            
            
            double dist = norm(Mat_<float>(scnPoint) - H * Mat_<float>(objPoint));
            // cout << "dist " << dist << endl;
            
//            dist = sqrt(pow(iP.x - candidate.x, 2) +
//                        pow(iP.y - candidate.y, 2));
            if (dist < threshDist) {
                inliers.push_back(i);
                num++;
            }
            
        }
        
        // cout << "num/numSamples " << num/numSamples << endl;
        if (num/numSamples > bestRatio) {
            bestRatio = num/numSamples;
            bestH = H;
        }
        
        if (num > numSamples * ransacRatio) {
            // cout << "Found good enough model" << endl;
            break;
        }
    }
    
    // Since we cannot return a tuple, we will encode the slope
    // and intercept as a Point2f(bestM, bestC).
    
    // cout << bestH << endl;
    return bestH;
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

Mat testRansac() {
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
    
    // cout << "m: " << m << ", c: " << c << endl;
    
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
    // imshow(plot, plotImg);
    imwrite("results/meanLine.JPG", plotImg);
    return plotImg;
}

void preprocess() {
    Stitcher::Mode mode = Stitcher::PANORAMA;
    vector<Mat> imgs;
    int panoStart = 26; // Actual start is 26
    int panoEnd = 60; // Actual end is 60

    for (int i = panoStart; i <= panoEnd - 3; i+=3) {
     vector<Mat> tempPanos;
     Mat img1 = imread("pano2/IMG_00" + to_string(i) + ".JPG");
     Mat img2 = imread("pano2/IMG_00" + to_string(i+1) + ".JPG");
     Mat img3 = imread("pano2/IMG_00" + to_string(i+2) + ".JPG");
     tempPanos.push_back(img1);
     tempPanos.push_back(img2);
     tempPanos.push_back(img3);
     Mat pano;     
     Ptr<Stitcher> stitcher = Stitcher::create(mode, true); 
     Stitcher::Status status = stitcher->stitch(tempPanos, pano);
     if (status != Stitcher::OK)
         cout << "Couldn't stitch images\n";
     imwrite("stitched/STITCHED_00" + to_string(i) + ".JPG", pano);
    }
}

Mat getPano() {
    Stitcher::Mode mode = Stitcher::PANORAMA;
    vector<Mat> imgs;

    int panoStart = 26; // Actual start is 26
    int panoEnd = 60; // Actual end is 60

    // preprocess();

    // For some reason, the preprocessing and the stitching of the 
    // preprocessed images can't be run at the same time.
    // So first, comment everything in the PART2 section and run
    // the homographie file, then uncomment it, and comment the 
    // call to the preprocess().

    // PART 2
    for (int i = panoStart; i <= panoEnd - 6; i+=3) {
        Mat img = imread("stitched/STITCHED_00" + to_string(i) + ".JPG");
        imgs.push_back(img);
    }

    // imshow("lastImg", imgs.at(imgs.size()-1));
    // waitKey();

    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(mode, false);
    Stitcher::Status status = stitcher->stitch(imgs, pano);
    if (status != Stitcher::OK)
        cout << "Couldn't stitch images\n";
    imwrite("results/finalPano.JPG", pano);
    // imshow("Pano", pano);
    // waitKey();
    // END PART 2

    return pano;
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
