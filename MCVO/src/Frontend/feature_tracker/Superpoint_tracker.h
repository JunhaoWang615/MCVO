#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#pragma once
#include <queue>

#include <memory>
#include <string>

#include <typeinfo>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <bits/stdc++.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "super_include/torch_cpp.hpp"

#include <Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "../../utility/tic_toc.h"

#include <ros/ros.h>

#include "../../utility/utility.h"
#include "../../utility/Twist.h"

#include "../sensors.h"

#include "trackerbase.h"

#define USE_ORB_SLAM2_DETECTOR 1

using namespace std;
using namespace camodocal;
using namespace Eigen;

// bool inBorder(const cv::Point2f &pt);
namespace MCVO
{
    // void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
    // void reduceVector(vector<int> &v, vector<uchar> status);

    class SupFeatureTracker : public TrackerBase
    {
    public:
        SupFeatureTracker();

        // void readImage(const cv::Mat &_img, const cv::Mat &_depth, double _cur_time);



        void readImage(const cv::Mat &_img, double _cur_time);

        void keypoints(cv::Mat img, int num);

        void setMask();

        void addPoints();

        bool updateID(unsigned int i);

        void readIntrinsicParameter(const string &calib_file);

        void showUndistortion(const string &name);

        void rejectWithF();

        void undistortedPoints();
        
        bool inBorder(const cv::Point2f &pt);

        void getPt(int idx, int &id, geometry_msgs::Point32 &p, cv::Point2f &p_uv, cv::Point2f &v);

        void getCurPt(int idx, cv::Point2f &cur_pt);

        void SortScore(vector<cv::KeyPoint> _pts, cv::Mat _desc, int num);

        void reduceMat(cv::Mat &mat, vector<uchar> &status);
        void reduceVector(vector<cv::KeyPoint> &v, vector<uchar> &status);



        cv::Mat mask;
        cv::Mat fisheye_mask;
        cv::Mat n_desc;
        cv::Mat prev_img, cur_img, forw_img;
        // cv::Mat prev_depth, cur_depth, forw_depth;
        vector<cv::KeyPoint> n_pts;
        vector<cv::KeyPoint> prev_pts, cur_pts, forw_pts;
        cv::Mat prev_desc, cur_desc, forw_desc;
        vector<cv::Point2f> prev_un_pts, cur_un_pts;
        vector<cv::Point2f> pts_velocity;
        vector<int> idx_vector;
        /* move to tracker base
        vector<int> ids;
        vector<int> track_cnt;
        camodocal::CameraPtr m_camera;
        int n_id;
        */
        map<int, cv::Point2f> cur_un_pts_map;
        map<int, cv::Point2f> prev_un_pts_map;

        bool useOpticalFlow = true;
        
        double cur_time;
        double prev_time;

        torch::jit::script::Module model;
        cv::Ptr<_cv::SuperGlue> superGlue;
        cv::Ptr<cv::Feature2D> superPoint;
        cv::Ptr<cv::Feature2D> superPoint_pin;
        float EPSILON;

        std::shared_ptr<MCVO::MCVOcamera> cam;
        
    };
} // namespace MCVO