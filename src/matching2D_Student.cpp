#include <numeric>
#include <map>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void KeyPointProcessor::matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    //std::cout<<"Starting matching"<<std::endl;
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
      int normType = cv::NORM_L2;
      if(descriptorType.compare("DES_BINARY") == 0)
      {
        normType = cv::NORM_HAMMING;
      }
      matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
      matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource,descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
      std::vector<std::vector<cv::DMatch> > kmatches;
      matcher->knnMatch(descSource,descRef, kmatches,2);
      for(auto per_desc_matches : kmatches)
      {
        if(per_desc_matches[0].distance <= 0.8 * per_desc_matches[1].distance)
        {
          matches.push_back(per_desc_matches[0]);
        }
      }
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void KeyPointProcessor::detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
}

void KeyPointProcessor::detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
  int blockSize = 2;
  int apertureSize = 3;
  int minResponse = 100;
  double k = 0.04;
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  cv::normalize(dst, dst_norm_scaled, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

  //std::cout<<"Running Harris detector image size : "<<dst_norm_scaled.rows<<" by "<<dst_norm_scaled.cols<<std::endl;
  for(auto  point = dst_norm_scaled.begin<float>(); point < dst_norm_scaled.end<float>(); point++)
  {
    if(*point > minResponse)
    {
      cv::Point2f candidatePoint(point.pos());
      cv::KeyPoint candidateKeyPoint(candidatePoint, sqrt(blockSize));
      bool candidateValid = true;

      for(auto keypoint : keypoints)
      {
        if(cv::KeyPoint::overlap(keypoint, candidateKeyPoint) != 0)
        {
          candidateValid = false;
          break;
        }
      }

      if(candidateValid)
      {
        //std::cout<<"Added keypoint with intensity "<< *point <<" at "<< candidateKeyPoint.pt.x<< ","<<candidateKeyPoint.pt.y<<std::endl;
        keypoints.push_back(candidateKeyPoint);
      }
    }
  }
}

void KeyPointProcessor::detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
  auto detector = cv::FastFeatureDetector::create();
  detector->detect(img, keypoints);
}
void KeyPointProcessor::detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
  auto detector = cv::BRISK::create();
  detector->detect(img, keypoints);
}
void KeyPointProcessor::detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
  auto detector = cv::ORB::create();
  detector->detect(img, keypoints);
}
void KeyPointProcessor::detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
  auto detector = cv::AKAZE::create();
  detector->detect(img, keypoints);
}
void KeyPointProcessor::detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
  auto detector = cv::xfeatures2d::SIFT::create();
  detector->detect(img, keypoints);
}
