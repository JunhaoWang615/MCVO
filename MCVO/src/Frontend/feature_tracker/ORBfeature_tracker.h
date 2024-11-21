#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
// #include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "../../utility/tic_toc.h"

#include <ros/ros.h>

#include "../../utility/utility.h"
#include "../../utility/Twist.h"

#include "../sensors.h"

#include "trackerbase.h"

#include "ORBextractor/ORBextractor.h"

#define USE_ORB_SLAM2_DETECTOR 1

using namespace std;
using namespace camodocal;
using namespace Eigen;

namespace MCVO
{
  class ORBFeatureTracker : public TrackerBase
  {
  public:
    ORBFeatureTracker();

    void readImage(const cv::Mat &_img, double _cur_time);

    void setMask();

    void addPoints(int cols, int rows);

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    bool inBorder(const cv::Point2f &pt);

    void getPt(int idx, int &id, geometry_msgs::Point32 &p, cv::Point2f &p_uv, cv::Point2f &v);

    void getCurPt(int idx, cv::Point2f &cur_pt);

    void reduceVector(vector<cv::KeyPoint> &v, vector<uchar> &status);
    void reduceMat1(cv::Mat &mat, vector<uchar> &status);
    void reduceMat(cv::Mat &mat, vector<uchar> &status);
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::KeyPoint> n_pts;
    cv::Mat n_desc;
    vector<cv::KeyPoint> prev_pts, cur_pts, forw_pts;
    cv::Mat prev_desc, cur_desc, forw_desc;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;

    struct Node
    {
        pair<cv::Point2f, cv::Point2f> Size;
        vector<int> Point_Lists;
    };

    vector<Node> node_List;
    vector<int> List;
    vector<vector<int>> listid_List;
    Node node;

    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    double cur_time;
    double prev_time;
#if USE_ORB_SLAM2_DETECTOR
    // Use ORB extractor provided in ORB_SLAM2
    std::shared_ptr<ORB_SLAM2::ORBextractor> ORBextractor_;
#else
    cv::Ptr<cv::ORB> ORBdetector;
    int nFeatures;
    float scaleFactor;
    int nlevels;
    int edgeThreshold;
    int firstLevel;
    int WTA_K;
    int scoreType;
    int patchSize;
    int fastThreshold;
#endif

    bool useOpticalFlow = true;
    // cv::BFMatcher matcher;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    int max_dist;

    std::shared_ptr<MCVO::MCVOcamera> cam;
  };
}