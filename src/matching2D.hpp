#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include <map>
#include <vector>
#include <typeindex>



void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

class KeyPointDetectors
{
  private:
    std::map<std::string, void (KeyPointDetectors::*) (std::vector<cv::KeyPoint> &, cv::Mat &)> _detMap;
    std::map<std::string, cv::Ptr<cv::DescriptorExtractor> (KeyPointDetectors::*) (int, int, float)> _descMap;
  public:
    KeyPointDetectors()
    {
      _detMap["SHITOMASI"] = &KeyPointDetectors::detKeypointsShiTomasi;
      _detMap["HARRIS"]    = &KeyPointDetectors::detKeypointsHarris;
      _detMap["FAST"]      = &KeyPointDetectors::detKeypointsFast;
      _detMap["BRISK"]     = &KeyPointDetectors::detKeypointsBrisk;
      _detMap["ORB"]       = &KeyPointDetectors::detKeypointsOrb;
      _detMap["AKAZE"]     = &KeyPointDetectors::detKeypointsAkaze;
      _detMap["SIFT"]      = &KeyPointDetectors::detKeypointsSift;


      _descMap["BRISK"] = &KeyPointDetectors::descKeypointsBriskExtractor;
      _descMap["BRIEF"] = &KeyPointDetectors::descKeypointsBriefExtractor;
      _descMap["ORB"]   = &KeyPointDetectors::descKeypointsOrbExtractor;
      _descMap["FREAK"] = &KeyPointDetectors::descKeypointsFreakExtractor;
      _descMap["AKAZE"] = &KeyPointDetectors::descKeypointsAkazeExtractor;
      _descMap["SIFT"]  = &KeyPointDetectors::descKeypointsSiftExtractor;
    }
    void visualize(std::string detectorType, std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
    {
      cv::Mat visImage = img.clone();
      cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      std::string windowName = detectorType + " Corner Detector Results";
      cv::namedWindow(windowName, 6);
      imshow(windowName, visImage);
      cv::waitKey(0);
    }

    void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
    {
      if(_descMap.count(descriptorType) != 0)
      {
        if(keypoints.size() > 0)
        {
          auto extractor = (this->*_descMap[descriptorType])(30, 3, 1.0);
          double t = (double)cv::getTickCount();
          extractor->compute(img, keypoints, descriptors);
          t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
          //std::cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;
        }
      }
      else
      {
        std::cout<<"ERROR!!! : Descriptor type "<< descriptorType <<" not supported!!"<<std::endl;
      }
    }
    cv::Ptr<cv::DescriptorExtractor> descKeypointsSiftExtractor(int threshold, int octaves, float patternScale)
    {
      return cv::xfeatures2d::SIFT::create();
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsFreakExtractor(int threshold, int octaves, float patternScale)
    {
      return cv::xfeatures2d::FREAK::create();
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsOrbExtractor(int threshold, int octaves, float patternScale)
    {
      return cv::ORB::create();
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsBriskExtractor(int threshold, int octaves, float patternScale)
    {
      return cv::BRISK::create(threshold, octaves, patternScale);
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsAkazeExtractor(int threshold, int octaves, float patternScale)
    {
      return cv::AKAZE::create();
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsBriefExtractor(int threshold, int octaves, float patternScale)
    {
      return cv::xfeatures2d::BriefDescriptorExtractor::create();
    }


    void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
    void detKeypoints(std::string detectorType, std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
    {

      if(_detMap.count(detectorType) != 0)
      {
        double t = (double)cv::getTickCount();
        (this->*_detMap[detectorType])(keypoints, img);
        if(bVis)
        {
          visualize(detectorType, keypoints, img);
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //std::cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;
      }
      else
      {
        std::cout<<"ERROR " << detectorType<<" Not supported!!" <<std::endl;
      }
    }

    void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                          std::vector<cv::KeyPoint> &kPtsRef,
                          cv::Mat &descSource,
                          cv::Mat &descRef,
                          std::vector<cv::DMatch> &matches,
                          std::string descriptorType,
                          std::string matcherType,
                          std::string selectorType);
};
#endif /* matching2D_hpp */
