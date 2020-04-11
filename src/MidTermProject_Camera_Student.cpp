/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
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
#include "matching2D.hpp"
#include <deque>
#include <algorithm>

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
  std::vector<string> detectors_1   = {"SHITOMASI","HARRIS","FAST","BRISK","ORB"};
  std::vector<string> descriptors_1 = {"BRISK", "BRIEF", "ORB", "FREAK", "SIFT"};
  std::vector<string> detectors_2   = {"AKAZE"};
  std::vector<string> descriptors_2 = {"AKAZE"};
  std::vector<string> detectors_3   = {"SIFT"};
  std::vector<string> descriptors_3 = {"BRISK", "BRIEF", "FREAK", "SIFT"};

  std::vector<std::pair<std::vector<string>,std::vector<string> > > det_desc_groups;
  det_desc_groups.push_back(std::make_pair(detectors_1, descriptors_1));
  det_desc_groups.push_back(std::make_pair(detectors_2, descriptors_2));
  det_desc_groups.push_back(std::make_pair(detectors_3, descriptors_3));

  ofstream f1;
  f1.open("./results.csv");
  f1<<"Detector,Descriptor,image_num,num_keypoints,num_matches,time_take,keyp_by_matches"<<endl;
  for(auto det_desc_group : det_desc_groups)
  for(auto detectorType : det_desc_group.first)
  {
    for(auto descriptorType : det_desc_group.second)
    {
      std::cout<<"Trying detector "<<detectorType<<" with descriptor "<<descriptorType<<std::endl;
      /* INIT VARIABLES AND DATA STRUCTURES */
      // data location
      string dataPath = "../";

      // camera
      string imgBasePath = dataPath + "images/";
      string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
      string imgFileType = ".png";
      int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
      int imgEndIndex = 9;   // last file index to load
      int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

      // misc
      int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
      deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
      bool bVis = false;            // visualize results
      KeyPointProcessor keyPointDetector;

      /* MAIN LOOP OVER ALL IMAGES */
      for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
      {
        double t = (double)cv::getTickCount();
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if(dataBuffer.size() >= dataBufferSize)
        {
          dataBuffer.pop_front();
        }
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        //cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        keyPointDetector.detKeypoints(detectorType, keypoints, imgGray, false);
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
          std::vector<cv::KeyPoint> inFocusKeyPoints;
          for(auto keypoint : keypoints)
          {
            if(vehicleRect.contains(keypoint.pt))
            {
              inFocusKeyPoints.push_back(keypoint);
            }
          }
          keypoints = inFocusKeyPoints;
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
          int maxKeypoints = 50;

          if (detectorType.compare("SHITOMASI") == 0)
          { // there is no response info, so keep the first 50 as they are sorted in descending quality order
            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
          }
          cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
          //cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        //cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        keyPointDetector.descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        //cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

          /* MATCH KEYPOINT DESCRIPTORS */

          vector<cv::DMatch> matches;
          string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
          string descriptorMatchType = "DES_BINARY"; // DES_BINARY, DES_HOG
          if(descriptorType.compare("SIFT") == 0)
          {
            descriptorMatchType = "DES_HOG";
          }
          string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

          //// STUDENT ASSIGNMENT
          //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
          //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

          keyPointDetector.matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
              (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
              matches, descriptorMatchType, matcherType, selectorType);

          //// EOF STUDENT ASSIGNMENT

          // store matches in current data frame
          (dataBuffer.end() - 1)->kptMatches = matches;

          //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

          // visualize matches between current and previous image
          bVis = false;
          if (bVis)
          {
            cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
            cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                matches, matchImg,
                cv::Scalar::all(-1), cv::Scalar::all(-1),
                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            string windowName = "Matching keypoints between two camera images";
            cv::namedWindow(windowName, 7);
            cv::imshow(windowName, matchImg);
            cout << "Press key to continue to next image" << endl;
            cv::waitKey(0); // wait for key to be pressed
          }
          bVis = false;
          t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
          std::cout<<"Num Keypoints : "<<keypoints.size()<<" num matches : "<<matches.size()<<" Time taken : "<<1000 * t/1.0<<"ms"<<std::endl;
          f1<<detectorType <<","
            <<descriptorType <<","
            <<imgIndex <<","
            <<keypoints.size() <<","
            <<matches.size() <<","
            <<1000*t/1.0 <<","
            <<std::min(1.0,(double)(matches.size())/(double)(keypoints.size()))
            <<std::endl;
        }

      } // eof loop over all images
    }
  }
  f1.close();

  return 0;
}
