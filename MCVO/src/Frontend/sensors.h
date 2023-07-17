#ifndef SENSORS_H
#define SENSORS_H
#include <mutex>
#include <queue>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <string>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "../utility/CameraPoseVisualization.h"
#include "MCVOfrontend_data.h"
#include <map>
#include <Eigen/Dense>

#define SHOW_LOG_DEBUG 0

using namespace std;
typedef pcl::PointXYZ PointType;
namespace MCVO
{
    enum sensor_type
    {
        MONOCULAR,
        STEREO,
        RGBD,
        THERMAL,
        LIDAR,
        UNKNOWN
    };
    enum detector_type
    {
        ShiTomasi,
        ORB,
        SPHORB,
        Superpoint
    };
    struct CameraProcessingResults
    {
        double timestamp;
        FeatureTrackerResults features;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
        cv::Mat image;
    };
    struct SyncCameraProcessingResults
    {
        double sync_timestamp;
        std::vector<std::shared_ptr<CameraProcessingResults>> results;
        std::vector<bool> isEmpty;
    };
    class FrontEndResultsSynchronizer
    {
    public:
        FrontEndResultsSynchronizer();
        ~FrontEndResultsSynchronizer(){};
        bool Sync();
        void addPool(std::string cam_name);
        void resize(size_t size);
        double timestamp;

        typedef std::shared_ptr<FrontEndResultsSynchronizer> Ptr;
        std::vector<std::shared_ptr<std::mutex>> result_mutexes;
        std::vector<std::shared_ptr<std::queue<std::shared_ptr<CameraProcessingResults>>>> results;
        // std::vector<std::pair<string, std::shared_ptr<CameraProcessingResults>>> current_result;
        unordered_map<string, int> tracker_tag;
        std::queue<std::shared_ptr<SyncCameraProcessingResults>> sync_results;
    };

    class MCVOsensor
    {
    public:
        MCVOsensor(sensor_type stype,
                    string topic,
                    string name,
                    ros::NodeHandle *node,
                    Eigen::Matrix3d R,
                    Eigen::Vector3d T);
        virtual ~MCVOsensor(){};

        // virtual void* ptr() = 0;

        sensor_type type;
        string topic, name;
        ros::Subscriber sub;
        ros::NodeHandle *frontend_node;
        Eigen::Matrix3d ext_R; // extrinsic rotation
        Eigen::Vector3d ext_T; // extrinsic translation
    };

    // Monocular
    // RGBD
    // Thermal
    class MCVOcamera : public MCVOsensor
    {
    public:
        MCVOcamera(sensor_type type,
                    string topic,
                    string name,
                    ros::NodeHandle *node,
                    Eigen::Matrix3d R,
                    Eigen::Vector3d T,
                    double fx, double fy, double cx, double cy, bool fisheye,
                    int w, int h);
        virtual ~MCVOcamera(){};
        // MCVOcamera* ptr(){return this;};
        bool setFisheye(string fisheye_path);
        void init_visualization();
        // P: imu P
        // R: imu R
        void pub_cam(Eigen::Vector3d & P, Eigen::Matrix3d & R, std_msgs::Header &header);

        bool FISHEYE;
        double fx, fy, cx, cy;
        int ROW, COL;
        cv::Mat mask;

        int MAX_CNT = 200;
        int MIN_DIST = 20;
        int FREQ = 10;

        double F_THRESHOLD = 1.0;
        bool EQUALIZE = 1;
        bool USE_VPI;
        int tracker_idx;

        bool first_image_flag = false;
        double first_image_time, last_image_time;
        int pub_count = 0;
        bool PUB_THIS_FRAME = false;
        int init_pub = 0;

        // for lidar
        Eigen::Matrix3d cam_laser_R;
        Eigen::Vector3d cam_laser_T;

        pcl::PointCloud<pcl::PointXYZ>::Ptr syncCloud = nullptr;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr imageCloud = nullptr;
        pcl::PointCloud<pcl::PointXYZI> normalize_point;
        pcl::KdTree<pcl::PointXYZI>::Ptr kdTree_ = nullptr;
        vector<vector<PointType>> pointsArray;
        // std::mutex lidar_mutex, data_mutex;
        cv::Mat depth_img;

        ros::Publisher pub_match, pub_depthimage;

        // For visualization 
        bool visualize = true;
        ros::Publisher pub_cam_pose, pub_cam_pose_visual, pub_slidewindow_camera_pose;
        CameraPoseVisualization cameraposevisual, keyframebasevisual;
        detector_type detector_type_ = ShiTomasi;
    };

    class MCVOlidar : public MCVOsensor
    {
    public:
        MCVOlidar(sensor_type type,
                   string topic,
                   string name,
                   ros::NodeHandle *node,
                   Eigen::Matrix3d R,
                   Eigen::Vector3d T,
                   int laser_type);
        virtual ~MCVOlidar(){};
        // MCVOlidar* ptr(){return this;};
        // int sensor_type_;
        int laser_type;
        double lidar_search_radius = 0.1;
        int max_level_ = 0;
        string lio_laser_frame, laser_frame, lio_world_frame, laser_frame_id;
    };

    class MCVOstereo : public MCVOsensor
    {
    public:
        MCVOstereo(sensor_type type,
                    string left_img,
                    string right_img,
                    string name,
                    ros::NodeHandle *node,
                    Eigen::Matrix3d left_R,
                    Eigen::Vector3d left_T,
                    Eigen::Matrix3d right_R,
                    Eigen::Vector3d right_T,
                    double lfx, double lfy, double lcx, double lcy,
                    double rfx, double rfy, double rcx, double rcy,
                    bool lFISHEYE, bool rFISHEYE,
                    int lw, int lh, int rw, int rh);

        virtual ~MCVOstereo(){};
        // MCVOstereo* ptr(){return this};
        bool setFisheye(string l_fisheye_path, string r_fisheye_path);

        bool lFISHEYE, rFISHEYE;
        double lfx, lfy, lcx, lcy;
        double rfx, rfy, rcx, rcy;
        int lcol, lrow, rcol, rrow;
        Eigen::Matrix3d lR, rR;
        Eigen::Vector3d lT, rT;
    };
} // namespace MCVO
#endif // SENSORS_H