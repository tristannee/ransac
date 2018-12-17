#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vector>

#include <iostream>

#include "image.h"
// #include "homographi e.hpp"

using namespace std;
using namespace cv;

int main()
{
	Image<uchar> I1 = Image<uchar>(imread("pano1/image0006.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	Image<uchar> I2 = Image<uchar>(imread("pano1/image0007.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	
	// namedWindow("I1", 1);
	// namedWindow("I2", 1);
	// imshow("I1", I1);
	// imshow("I2", I2);

	//Ptr<AKAZE> D = AKAZE::create();
    Ptr<AKAZE> akaze = AKAZE::create();
	// ...
	//vector<KeyPoint>
    
    vector<KeyPoint> m1, m2;
	// ...
    
    Mat d1, d2;
    
    akaze->detectAndCompute(I1, noArray(), m1, d1);
    akaze->detectAndCompute(I2, noArray(), m2, d2);
    
	
	//Mat J;
	//drawKeypoints(...
	
    
    Mat J;
    drawKeypoints(I1, m1, J);
    // imshow("J", J);

	//BFMatcher M ...
    
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(d1, d2, nn_matches, 2);
    
    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    double min = nn_matches[0][0].distance;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        
        if(dist1 < 0.8f * dist2) {
            if(dist1<min){
                min = dist1;
            }
            matched1.push_back(m1[first.queryIdx]);
            matched2.push_back(m2[first.trainIdx]);
            good_matches.push_back(first);
        }
    }
    
    vector<DMatch> good_matches2;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        
        if(dist1 < 10*min) {
            
            good_matches2.push_back(first);
        }
    }

	// drawMatches ...
	
    Mat match;
    drawMatches(I1, m1, I2, m2, good_matches2, match);
    // imshow("match", match);
      
	// Mat H = findHomography(...
    
    vector<Point2f> obj;
    vector<Point2f> scene;
    for( int i = 0; i < good_matches2.size(); i++ )
    {
        obj.push_back( m1[ good_matches2[i].queryIdx ].pt );
        scene.push_back( m2[ good_matches2[i].trainIdx ].pt );
    }
    
//    cout << obj << endl;
//    cout << scene << endl;
    
    Mat H1 = findHomography( scene, obj, CV_RANSAC );
    
    // cout << "H1" << H1;
    
    Mat H = ransacGeneralAlternative(scene, obj);
    
    // cout << "H" << H;
    

	// Mat K(2 * I1.cols, I1.rows, CV_8U);
	// warpPerspective( ...
    
    Mat K(2 * I1.cols, I1.rows, CV_8U);
    Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
    warpAffine(I1, K, trans_mat, Size( 2*I1.cols, I1.rows));
    Mat K2;
    warpPerspective(I2, K2, H, K.size());
    
   
    for (int i = 0; i < 490 ; i++) {
        for (int j = 0; j < 1400; j++) {
            if (K.at<float>(i,j) == 0){
                // cout<<i<<" "<<j<<endl;
                K.at<float>(i,j) = K2.at<float>(Point(j,i));
            }
        }
    }
    
    // imshow("I1I2", K) ;
    
    
    Image<uchar> I3 = Image<uchar>(imread("pano1/image0008.jpg", CV_LOAD_IMAGE_GRAYSCALE));
    
    //vector<KeyPoint> m1, m2;
    // ...
    
    //Mat d1, d2;
    
    akaze->detectAndCompute(K, noArray(), m1, d1);
    akaze->detectAndCompute(I3, noArray(), m2, d2);
    
    
    //Mat J;
    //drawKeypoints(...
    
    
   // Mat J;
    drawKeypoints(K, m1, J);
    // imshow("J", J);
    
    //BFMatcher M ...
    
    //BFMatcher matcher(NORM_HAMMING);
    //vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(d1, d2, nn_matches, 2);
    
//    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    good_matches.clear();
    matched1.clear();
    matched2.clear();
    min = nn_matches[0][0].distance;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        
        if(dist1 < 0.8f * dist2) {
            if(dist1<min){
                min = dist1;
            }
            matched1.push_back(m1[first.queryIdx]);
            matched2.push_back(m2[first.trainIdx]);
            good_matches.push_back(first);
        }
    }
    
    good_matches2.clear();
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        
        if(dist1 < 10*min) {
            
            good_matches2.push_back(first);
        }
    }
    
    
    // drawMatches ...
    
    
    drawMatches(K, m1, I3, m2, good_matches2, match);
    // imshow("match", match);
    
    
    // Mat H = findHomography(...
    
    
    obj.clear();
    scene.clear();
    for( int i = 0; i < good_matches2.size(); i++ )
    {
        obj.push_back( m1[ good_matches2[i].queryIdx ].pt );
        scene.push_back( m2[ good_matches2[i].trainIdx ].pt );
    }
    
    //    cout << obj << endl;
    //    cout << scene << endl;
    
    
    
    H1 = findHomography( scene, obj, CV_RANSAC );
    
    // cout << "H1" << H1;
    
    H = ransacGeneralAlternative(scene, obj);
    Mat Hinv = ransacGeneralAlternative(obj, scene);
    
    // cout << "H" << H;
    
    
    // Mat K(2 * I1.cols, I1.rows, CV_8U);
    // warpPerspective( ...
    
    Mat Kb(K.cols, 2 * K.rows, CV_8U);
    //Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
//    warpAffine(K, Kb, trans_mat, Size( K.cols, 2 * K.rows));
    warpPerspective(K, Kb, Hinv, Size( K.cols, 2 * K.rows));

    // imshow("I1I2kb", Kb) ;
    Mat K2b;
    //warpPerspective(I3, K2b, H, Kb.size());
    warpAffine(I3, K2b, trans_mat, Size( K.cols, 2 * K.rows));
    // imshow("I1I2kb2", K2b) ;
    
    
    for (int i = 0; i < 532 ; i++) {
        for (int j = 0; j < 1400; j++) {
            if (Kb.at<float>(Point(j,i)) == 0){
                // cout<<i<<" "<<j<<endl;
                Kb.at<float>(Point(j,i)) = K2b.at<float>(Point(j,i));
            }
        }
    }
    
    // imshow("I1I2", Kb) ;
    imwrite("results/threeImagesHomography.JPG", Kb);

    // waitKey(0);

    Mat plot = testRansac();
    Mat pano = getPano();

    imshow("Mean Line Estimation", plot);
    waitKey(0);
    imshow("3 Images", Kb);
    waitKey(0);
    imshow("Panorama", pano);
    waitKey(0);

    return 0 ;
}
    

