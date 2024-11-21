#pragma once
#include "../MCVOfeature_manager.h"

#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "alignment_factor.h"
#include "../../utility/utility.h"
#include "../../utility/Twist.h"
#include "../../Utils/EigenTypes.h"
#include "../../Frontend/MCVOfrontend_data.h"
#include <pcl/point_cloud.h>

#include <ros/ros.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <iostream>
#include <map>

using namespace Eigen;
using namespace std;
#define USE_DEPTH_INITIAL 0
class ImageFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImageFrame(){};
    ImageFrame(const MCVO::SyncCameraProcessingResults &input)
    {
        points.resize(NUM_OF_CAM);
        Twi.resize(NUM_OF_CAM);
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            points[c] = input.results[c]->features;
        }
        timestamp = input.sync_timestamp;
    }

public:
    vector<MCVO::FeatureTrackerResults> points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
    cv::Mat image;
    IntegrationBase *pre_integration = nullptr;

    // laser odometry information
    Transformd Tw_imu_meas;
    Vector3d vel_imu_meas;
    Vector3d Ba_meas;
    Vector3d Bg_meas;
    double gravity_meas;
    int reset_id = -1; // to notifiy the status of laser odometry

    vector<Transformd> Twi;
    double timestamp;
    bool is_key_frame = false;
    bool laser_odom_vio_sync = false;
}; // class ImageFrame

bool VisualIMUAlignment(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x, int base_cam);
bool VisualIMUAlignmentWithDepth(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x, int base_cam);

void solveGyroscopeBias(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, int base_cam);

bool VisualIMUAlignmentSolver(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x, int base_cam);

// For two cameras
bool MultiCameraAlignment(Eigen::aligned_map<double, ImageFrame> &all_image_frame, int l, VectorXd &x);

// For multiple cameras
bool MultiCameraAlignment(Eigen::aligned_map<double, ImageFrame> &all_image_frame, int l, VectorXd &x, vector<bool> &state);