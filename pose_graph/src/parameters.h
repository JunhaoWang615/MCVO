#pragma once

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>

extern std::vector<camodocal::CameraPtr> m_camera_vector;
extern std::vector<Eigen::Vector3d> tic_vector;
extern std::vector<Eigen::Matrix3d> qic_vector;
extern ros::Publisher pub_match_img;
extern ros::Publisher pub_match_points;
extern int VISUALIZATION_SHIFT_X;
extern int VISUALIZATION_SHIFT_Y;
extern std::string BRIEF_PATTERN_FILE;
extern std::string POSE_GRAPH_SAVE_PATH;
extern int ROW;
extern int COL;
extern std::string VINS_RESULT_PATH;
extern int DEBUG_IMAGE;
extern int FAST_RELOCALIZATION;


