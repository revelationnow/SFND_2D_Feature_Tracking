# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## Results
1. The number of keypoints detected on each image for different combinations of detectors and descriptors is in keypoints.html
2. The number of matches detected on each image pair for different combinations of detectors and descriptors is in matches.html
3. The time taken for the different detector and descriptor pairs is in time_taken.html


# NUMBER OF KEYPOINTS
![](keypoints.svg?raw=true)

Clearly FAST appears to detect a lot more keypoints than the other detectors. BRISK comes in second place while the remaining appear to be close to each other in the number of points detected. The SHITOMASI and SIFT detectors have a much tighter bound than the others across the different images, indicating a possibility that they might be more robust, however these need to be tested over a much larger corpus of images to make that claim.


# NUMBER OF MATCHES
![](matches.svg?raw=true)

Here again FAST appears to have a lot more matches compared to other detectors, regardless of the descriptor type used. This is likely an artifact of having detected a large number of keypoints in the first place. If we take a ratio of number of keypoints to number of matches, we see this metric is very similar between detectors and descriptors.

![](keyp_by_matches.svg?raw=true)

The above image normalizes the number of matches by keypoints. Here we see that the spread of the matches per keypoint is very diverse. The Harris detector for instance is very easily thrown off. AKAZE appears to be the most robust here. FAST has a large number of keypoints and matches, however the quality of the matches appears to be lower compared to AKAZE, but it is still pretty close when used with a SIFT descriptor. This is true for ORB detectors with SIFT descriptors as well. SIFT and
BRISK are the next in line.


# TIME TAKEN
![](time_taken.svg?raw=true)

From a time taken perspective, FAST appears to be the fastest, followed by ORB and SHITOMASI, subsequently followed by SIFT and HARRIS. Here the choice of descriptor appears to affect the results a lot. BRISK despite its name is a pretty slow choice, all the detectors show their slowest performance using BRISK. BRISK is also not the best in the matches per keypoint metric.

# FINAL CONCLUSION

The Top 3 Detector/Descriptor pairs based on a combination of number of keypoints, quality of keypoints and time taken are:

1. FAST Detector with SIFT Descriptors
2. ORB Detector with SIFT Descriptors
3. AKAZE Detector with AKAZE Descriptors
