#ifndef MCVOFRONTEND_H
#define MCVOFRONTEND_H

#include <mutex>
#include <queue>
#include <unordered_map>
#include <string>
#include <limits>
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

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

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "MCVOfrontend_data.h"
#include "feature_tracker/trackerbase.h"
#include "feature_tracker/feature_tracker.h"
#include "feature_tracker/vpi_feature_tracker.h"
#include "feature_tracker/ORBfeature_tracker.h"
#include "feature_tracker/SPHORBfeature_tracker.h"
#include "feature_tracker/Superpoint_tracker.h"

#include "sensors.h"
#include "../utility/CameraPoseVisualization.h"



// typedef pcl::PointXYZ PointType;
#define SINGLEDEPTH 0
#define USE_LIDAT_DEPTH 1
#define fuse_global_point 1
#define MERGELASER 0

using namespace std;
namespace MCVO
{
    typedef std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, std::shared_ptr<MCVO::SyncCameraProcessingResults>>>
        imuAndFrontEndMeasurement;

    typedef std::vector<std::shared_ptr<MCVO::SyncCameraProcessingResults>>
        FrontEndMeasurement;
    class MCVOfrontend
    {
    public:
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> syncPolicy;

    public:
        explicit MCVOfrontend(string config_file);
        static constexpr int N_SCAN = 64;
        static constexpr int Horizon_SCAN = 1000;
        static constexpr float ang_res_x = 0.2;
        static constexpr float ang_res_y = 0.427;
        static constexpr float ang_bottom = 24.9;
        static constexpr int groundScanInd = 16;
        static constexpr bool distortion_flag = false;
        static constexpr double distortion_img[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        static constexpr int Boundary = 4;
        static constexpr int num_bins = 360;
        static constexpr int WINDOW_SIZE = 10;

    public:
        void
        setUpROS(ros::NodeHandle *pub_node, ros::NodeHandle *private_node);

        void
        addSensors(cv::FileStorage &fsSettings, ros::NodeHandle *private_node);

        void
        addMonocular(cv::FileNode &fsSettings, ros::NodeHandle *private_node);
        // img_callback

        void
        addStereo(cv::FileNode &fsSettings, ros::NodeHandle *private_node);
        // stereo_img_callback
        void
        addInfrared(cv::FileNode &fsSettings, ros::NodeHandle *private_node);

        void
        addRGBD(cv::FileNode &fsSettings, ros::NodeHandle *private_node);

        void
        addThermal(cv::FileNode &fsSettings, ros::NodeHandle *private_node);

        void
        addlidar(cv::FileNode &fsSettings, ros::NodeHandle *private_node);
        // lidar_callback

        void readCamToLidarExtrinsic(cv::FileNode &fsSettings);

        void processImage(const sensor_msgs::ImageConstPtr &color_msg);

        void
        RGBD_callback(const sensor_msgs::ImageConstPtr &color_msg);

        // Lidar related functions
        // void
        // lidar_callback(const sensor_msgs::PointCloud2ConstPtr &laser_msg);
        void
        processLidar(const sensor_msgs::PointCloud2ConstPtr &laser_msg);

        void
        assign_feature_depth(cv::Point2f &p, double &depth, std::shared_ptr<MCVOcamera> const &cam);

        static Eigen::Vector2f
        xyz_to_uv(pcl::PointXYZ &xyz);

        Eigen::Vector2f
        xyz_to_uv(const Eigen::Vector3f &xyz);

        static bool
        is_in_image(const Eigen::Vector2d &uv, int boundary, float scale, std::shared_ptr<MCVOcamera> ptr);

        void
        show_image_with_points(std::shared_ptr<MCVOcamera> ptr, size_t num_level);

        void
        cloud_in_image(std::shared_ptr<MCVOcamera> ptr);

        void
        get_depth(const geometry_msgs::Point32 &features_2d, pcl::PointXYZI &p);

        bool
        create_sphere_cloud_kdtree(const ros::Time &stamp_cur, cv::Mat cur_image, const pcl::PointCloud<PointType>::Ptr &depth_cloud);

        void
        project_cloud_image(const ros::Time &stamp_cur, cv::Mat imageCur, const pcl::PointCloud<PointType>::Ptr &depth_cloud_local);

        float
        pointDistance(PointType p);

        void
        getColor(float p, float np, float &r, float &g, float &b);

        float
        pointDistance(PointType p1, PointType p2);
        // Lidar related functions

        template <typename T>
        sensor_msgs::PointCloud2
        publishCloud(ros::Publisher *thisPub, T thisCloud, ros::Time thisStamp, std::string thisFrame);

    public:
        ros::NodeHandle *pub_node_;
        ros::NodeHandle *private_node_;

        ros::Publisher pub_img, pub_match, pub_featuredepth, pub_depthimage;
        ros::Publisher pub_restart, pub_lidar_map, pub_depth_cloud, pub_depth_points;

#if SINGLEDEPTH
        message_filters::Subscriber<sensor_msgs::Image> sub_image;
        message_filters::Subscriber<sensor_msgs::PointCloud2> sub_laserscan;
        message_filters::Synchronizer<syncPolicy> sync{syncPolicy(10)};
#else
        ros::Subscriber sub_image;
        ros::Subscriber sub_laserscan;
#endif

    public:
        // data interface
        std::mutex *datamuex_ = nullptr;
        std::mutex lidar_mutex, lidar_aligned_image_mutex;
        std::queue<MCVO::FrontEndResult::Ptr> *fontend_output_queue = nullptr;

    public:
        // lidar, currently support 1
        // TODO: encapsulation

        pcl::PointCloud<pcl::PointXYZ>::Ptr syncCloud = nullptr;
        // pcl::PointCloud<pcl::PointXYZRGBA>::Ptr imageCloud = nullptr;

        // pcl::PointCloud<pcl::PointXYZI> normalize_point;
        // pcl::KdTree<pcl::PointXYZI>::Ptr kdTree_ = nullptr;

        // vector<vector<PointType>> pointsArray;

        // deque<pcl::PointCloud<PointType>> cloudQueue;
        // deque<double> timeQueue;
        // pcl::PointCloud<PointType>::Ptr depthCloud = nullptr;
        // pcl::PointCloud<pcl::PointXYZI>::Ptr depth_cloud_unit_sphere = nullptr;

        // tracker
        //
        vector<std::shared_ptr<TrackerBase>> trackerData;
        string lidar_name;
        int pub_count = 1;

        bool first_image_flag = true;
        double first_image_time;
        double last_image_time = 0;
        // double lidar_search_radius = 0.5;

        bool init_pub = false;
        bool create_kd_tree = false;
        vector<cv::Mat> laservisual_alligned_imgs_;
        // cv::Mat depth_img;

        // multicamera config
        int num_of_cam = 1;
        int SHOW_TRACK = 1;
        string config_file;
        bool has_lidar = false;
        int lidar_idx; // lidar index in sensors
        double lidar_search_radius = 0.5;
        int LASER_TYPE;
        vector<std::shared_ptr<MCVOsensor>> sensors;

        //<frame_id, <index in sensors, index in tracker_data>>, tracker data index also indexes in backend for quick access for data
        unordered_map<string, int> sensors_tag;
        unordered_map<string, int> tracker_tag;

        FrontEndResultsSynchronizer synchronizer;
    };
    inline void
    img_callback(const sensor_msgs::Image::ConstPtr color_msg, MCVOfrontend *frontend)
    {
        frontend->processImage(color_msg);
    };
    inline void
    lidar_callback(const sensor_msgs::PointCloud2::ConstPtr laser_msg, MCVOfrontend *frontend)
    {
        frontend->processLidar(laser_msg);
    };

    // Owned by System. Global variable to enable easy access from different modules.
    inline MCVOfrontend *MCVOfrontend_;
} // namespace MCVO
#endif // FRONTEND_H
