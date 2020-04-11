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

class KeyPointProcessor
{
  private:
    std::map<std::string, void (KeyPointProcessor::*) (std::vector<cv::KeyPoint> &, cv::Mat &)> _detMap;
    std::map<std::string, cv::Ptr<cv::DescriptorExtractor> (KeyPointProcessor::*) ()> _descMap;
  public:
    KeyPointProcessor()
    {
      _detMap["SHITOMASI"] = &KeyPointProcessor::detKeypointsShiTomasi;
      _detMap["HARRIS"]    = &KeyPointProcessor::detKeypointsHarris;
      _detMap["FAST"]      = &KeyPointProcessor::detKeypointsFast;
      _detMap["BRISK"]     = &KeyPointProcessor::detKeypointsBrisk;
      _detMap["ORB"]       = &KeyPointProcessor::detKeypointsOrb;
      _detMap["AKAZE"]     = &KeyPointProcessor::detKeypointsAkaze;
      _detMap["SIFT"]      = &KeyPointProcessor::detKeypointsSift;


      _descMap["BRISK"] = &KeyPointProcessor::descKeypointsBriskExtractor;
      _descMap["BRIEF"] = &KeyPointProcessor::descKeypointsBriefExtractor;
      _descMap["ORB"]   = &KeyPointProcessor::descKeypointsOrbExtractor;
      _descMap["FREAK"] = &KeyPointProcessor::descKeypointsFreakExtractor;
      _descMap["AKAZE"] = &KeyPointProcessor::descKeypointsAkazeExtractor;
      _descMap["SIFT"]  = &KeyPointProcessor::descKeypointsSiftExtractor;
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
          auto extractor = (this->*_descMap[descriptorType])();
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
    cv::Ptr<cv::DescriptorExtractor> descKeypointsSiftExtractor()
    {
      return cv::xfeatures2d::SIFT::create();
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsFreakExtractor()
    {
      return cv::xfeatures2d::FREAK::create();
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsOrbExtractor()
    {
      return cv::ORB::create();
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsBriskExtractor()
    {
      int threshold = 30;
      int octaves = 3;
      float patternScale = 1.0;
      return cv::BRISK::create(threshold, octaves, patternScale);
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsAkazeExtractor()
    {
      return cv::AKAZE::create();
    }

    cv::Ptr<cv::DescriptorExtractor> descKeypointsBriefExtractor()
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
