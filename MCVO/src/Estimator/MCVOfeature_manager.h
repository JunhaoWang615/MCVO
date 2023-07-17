#ifndef MCVOFEATURE_MANAGER_H
#define MCVOFEATURE_MANAGER_H
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "../Utils/EigenTypes.h"
#include "../Frontend/MCVOfrontend_data.h"
#include "../Frontend/MCVOfrontend.h"
#include "parameters.h"
#define MIN_DEPTH_MEASURED 0.1
#define MAX_DEPTH_MEASURED 100
using namespace MCVO;

class KeypointObservation
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeypointObservation() = default;
    KeypointObservation(const Eigen::Matrix<double, 8, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        depth_measured = _point(7);
        cur_td = td;
    }

    Vector3d point;        // normalize point
    Vector2d uv;           // image point
    Vector2d velocity;     // 相邻帧计算的特征点的速度
    double depth_measured; // depth getted from lidar
    double cur_td;

    // bool is_used;      // TODO: 这个好像没什么用
};

class KeyPointLandmark
{
public:
    enum class EstimateFlag
    {
        NOT_INITIAL,
        DIRECT_MEASURED,
        TRIANGULATE
    };

    enum class SolveFlag
    {
        NOT_SOLVE,
        SOLVE_SUCC,
        SOLVE_FAIL
    };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeyPointLandmark() = default;
    KeyPointLandmark(MCVO::FeatureID _feature_id, MCVO::TimeFrameId _start_frame, double _measured_depth)
        : feature_id(_feature_id),
          kf_id(_start_frame),
          start_frame(_start_frame), // TODO, start_frame 需要去掉
          used_num(0),
          estimated_depth(-1.0),
          measured_depth(_measured_depth),
          estimate_flag(EstimateFlag::NOT_INITIAL),
          solve_flag(SolveFlag::NOT_SOLVE)
    {
    }

public:
    MCVO::FeatureID feature_id; // landmark id
    MCVO::TimeFrameId kf_id;    // host frame id
    int start_frame;
    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    double measured_depth; // TODO:这个好像没什么用

    EstimateFlag estimate_flag; // 0 initial; 1 by depth image; 2 by triangulate
    //    EstimateFlag estimate_flag;
    SolveFlag solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
    Vector3d gt_p;
    // TODO: 使用这个代替
    Eigen::aligned_map<MCVO::TimeFrameId, KeypointObservation> obs;

    //　backend parameter interface
    std::array<double, SIZE_FEATURE> data;
};

class FeatureManager
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FeatureManager();
    FeatureManager(Matrix3d _Rs[],MCVOfrontend* frontend);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount(int c);

    bool addFeatureCheckParallax(int frame_count,
                                 const MCVO::TimeFrameId frame_id,
                                 const MCVO::SyncCameraProcessingResults &image,
                                 double td);

    Eigen::aligned_vector<Eigen::aligned_vector<pair<Vector3d, Vector3d>>>
    getCorresponding(int frame_count_l, int frame_count_r);

    Eigen::aligned_vector<Eigen::aligned_vector<pair<Vector3d, Vector3d>>>
    getCorrespondingWithDepth(int frame_count_l, int frame_count_r);

    void setCamState(vector<bool> cam_state);

    void
    triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[], int base_cam);

    void
    triangulateWithDepth(Vector3d Ps[], Vector3d tic[], Matrix3d ric[],int base_cam);

    // TODO 新加的函数,backend optimization interface
    //-------------------------
    void invDepth2Depth();
    void depth2InvDepth();
    void resetDepth();
    void removeFailures();
    //-------------------------

    // TODO: 重写某一帧观测的3d的函数
    //----------------------------

    /// \briefn 当滑窗开始优化时,边缘化一帧时, 3d点的主导帧,可能发生改变
    /// \param marg_frame_tid 要边缘化关键帧的id号
    /// \param Ps 滑窗中帧的position
    /// \param tic camera to imu transalation
    /// \param ric camera to imu roration
    void removeOneFrameObservationAndShiftDepth(MCVO::TimeFrameId marg_frame_tid,
                                                Vector3d Ps[],
                                                Vector3d tic[], Matrix3d ric[]);

    void
    removeOneFrameObservation(MCVO::TimeFrameId marg_frame_tid);

    //----------------------------
    // TODO: 重写某一帧观测的3d的函数

    // TODO: new initialization using depth
    //----------------------------
    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[],int base_cam);
    bool solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P,
                        vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    //----------------------------
public:
    // estimator interface
    std::set<double> *local_active_frames_ = nullptr;
    std::map<int, double> *int_frameid2_time_frameid_ = nullptr;
    std::map<double, int> *time_frameid2_int_frameid_ = nullptr;

public:
    std::vector<Eigen::aligned_unordered_map<MCVO::FeatureID, KeyPointLandmark>> KeyPointLandmarks;

    // int NUM_OF_CAM;
    std::vector<int> last_track_num;
    std::vector<bool> cam_state;

private:
    double compensatedParallax2(const KeyPointLandmark &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d* ric;
    // Matrix3d ric[NUM_OF_CAM];
    // For ric, tic
    MCVOfrontend* frontend_;
};

#endif // FEATURE_MANAGER_H
