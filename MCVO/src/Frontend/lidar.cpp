// Lidar related functions for MCVO
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include "MCVOfrontend.h"
#include "sensors.h"
using namespace MCVO;

// void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr laser_msg, MCVOfrontend *frontend)
// {
//     frontend->processLidar(laser_msg);
// }

void MCVOfrontend::assign_feature_depth(cv::Point2f &p, double &depth, std::shared_ptr<MCVOcamera> const &cam)
{
    // ROS_DEBUG("Into assign feature depth");

    if (!cam->kdTree_)
    {
        ROS_DEBUG("kd Tree not init");
        return;
    }
    if (!cam->kdTree_->getInputCloud())
    {
        ROS_DEBUG("kd Tree not set");
        return;
    }
    if (!cam->kdTree_->getInputCloud()->size())
    {
        ROS_DEBUG("Empty kd tree!");
        return;
    }
    // ROS_DEBUG("Into assign feature depth2, kd tree size: %d",kdTree_->getInputCloud()->size());
    std::array<float, 3> pts = {p.x, p.y, 1};

    pcl::PointXYZI P;
    // 将视觉的特征点归一化到10这个平面上
    P.x = 10 * p.x;
    P.y = 10 * p.y;
    P.z = 10;

    // 根据当前视觉的特征点，对当前激光点搜索最近临，然后确定深度值
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqrDis;
    pointSearchInd.clear();
    pointSearchSqrDis.clear();
    ROS_DEBUG("KD Tree search");
    try
    {
        // ROS_DEBUG("KD Tree search2");
        cam->kdTree_->nearestKSearch(P, 3, pointSearchInd, pointSearchSqrDis);
        // ROS_DEBUG("KD Tree search3");
    }
    catch (...)
    {
        ROS_ERROR_STREAM("Kd tree search incorrect");
        return;
    }
    ROS_DEBUG("Processing search results");
    double minDepth, maxDepth;
    // use lidar depth map to assign depth to feature
    // 如果搜索的距离小于threshold，并且能找到3个点
    if (pointSearchInd.size() == 0)
    {
        return;
    }
    if (pointSearchSqrDis[0] < lidar_search_radius && pointSearchInd.size() == 3)
    {
        // ROS_DEBUG("Normalized point -> 3D point");
        //取出该特征对应的激光点，恢复成真正的空间3D点
        pcl::PointXYZI depthPoint = cam->normalize_point.points[pointSearchInd[0]];
        double x1 = depthPoint.x * depthPoint.intensity / 10;
        double y1 = depthPoint.y * depthPoint.intensity / 10;
        double z1 = depthPoint.intensity;
        minDepth = z1;
        maxDepth = z1;

        depthPoint = cam->normalize_point.points[pointSearchInd[1]];
        double x2 = depthPoint.x * depthPoint.intensity / 10;
        double y2 = depthPoint.y * depthPoint.intensity / 10;
        double z2 = depthPoint.intensity;
        minDepth = (z2 < minDepth) ? z2 : minDepth;
        maxDepth = (z2 > maxDepth) ? z2 : maxDepth;

        depthPoint = cam->normalize_point.points[pointSearchInd[2]];
        double x3 = depthPoint.x * depthPoint.intensity / 10;
        double y3 = depthPoint.y * depthPoint.intensity / 10;
        double z3 = depthPoint.intensity;
        minDepth = (z3 < minDepth) ? z3 : minDepth;
        maxDepth = (z3 > maxDepth) ? z3 : maxDepth;

        double u = pts[0];
        double v = pts[1];

        //根据三个激光点插值出深度值
        pts[2] = (x1 * y2 * z3 - x1 * y3 * z2 - x2 * y1 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x3 * y2 * z1) / (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2 + u * y1 * z2 - u * y2 * z1 - v * x1 * z2 + v * x2 * z1 - u * y1 * z3 + u * y3 * z1 + v * x1 * z3 - v * x3 * z1 + u * y2 * z3 - u * y3 * z2 - v * x2 * z3 + v * x3 * z2);
        // 如果插值出的深度是无限的
        if (!std::isfinite(pts[2]))
        {
            pts[2] = z1;
        }
        if (maxDepth - minDepth > 2)
        { // 如果最大距离-最小距离>2 ，则融合距离失败
            pts[2] = -1;
        }
        else if (pts[2] - maxDepth > 0.2)
        { //如果融合距离比最大的距离大0.2，
            pts[2] = maxDepth;
        }
        else if (pts[2] - minDepth < -0.2)
        {                      //如果融合距离比最小的距离还小
            pts[2] = minDepth; //那么选最小距离
        }
    }
    else // use lidar depth map to successfully assign depth feature end
    {
        pts[2] = -1;
    } // cannot find 3 lidar points to assign depth feature end

    depth = pts[2];
    // ROS_DEBUG("Depth: %f",depth);
} // assign_feature_depth end

/*
Eigen::Vector2f
MCVOfrontend::xyz_to_uv(pcl::PointXYZ &xyz)
{
    float fx_ = fx;
    float fy_ = fy;
    float cx_ = cx;
    float cy_ = cy;
    //转换到图像平面点
    float x = fx_ * xyz.x + cx_ * xyz.z;
    float y = fy_ * xyz.y + cy_ * xyz.z;
    float z = xyz.z;

    //转换到图像uv
    Eigen::Vector2f uv(x / z, y / z);

    //如果没有distortion就直接返回uv
    if (!distortion_flag)
    {
        return uv;
    }
    else
    {
        // uv 去除图像畸变
        float xx = xyz.x / xyz.z;
        float yy = xyz.y / xyz.z;
        float r2 = xx * xx + yy * yy;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float a1 = 2 * xx * yy;
        float a2 = r2 + 2 * xx * xx;
        float a3 = r2 + 2 * yy * yy;
        float cdist =
            1 + (float)distortion_img[0] * r2 + (float)distortion_img[1] * r4 + (float)distortion_img[4] * r6;
        float xd = xx * cdist + (float)distortion_img[2] * a1 + (float)distortion_img[3] * a2;
        float yd = yy * cdist + (float)distortion_img[2] * a3 + (float)distortion_img[3] * a1;
        Eigen::Vector2f uv_undist(xd * (float)fx_ + (float)cx_, yd * (float)fy_ + (float)cy_);

        return uv_undist;
    }
} // function xyz to_uv end

Eigen::Vector2f
MCVOfrontend::xyz_to_uv(const Eigen::Vector3f &xyz)
{
    float fx_ = fx;
    float fy_ = fy;
    float cx_ = cx;
    float cy_ = cy;

    float x = fx_ * xyz(0) + cx_ * xyz(2);
    float y = fy_ * xyz(1) + cy_ * xyz(2);
    float z = xyz(2);
    Eigen::Vector2f uv(x / z, y / z);

    if (!distortion_flag)
    {
        return uv;
    }
    else
    {
        float xx = xyz(0) / xyz(2);
        float yy = xyz(1) / xyz(2);
        float r2 = xx * xx + yy * yy;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float a1 = 2 * xx * yy;
        float a2 = r2 + 2 * xx * xx;
        float a3 = r2 + 2 * yy * yy;
        float cdist =
            1 + (float)distortion_img[0] * r2 + (float)distortion_img[1] * r4 + (float)distortion_img[4] * r6;
        float xd = xx * cdist + (float)distortion_img[2] * a1 + (float)distortion_img[3] * a2;
        float yd = yy * cdist + (float)distortion_img[2] * a3 + (float)distortion_img[3] * a1;
        Eigen::Vector2f uv_undist(xd * (float)fx_ + (float)cx_, yd * (float)fy_ + (float)cy_);

        return uv_undist;
    }
} // function xyz_to_uv end
*/
bool MCVOfrontend::is_in_image(const Eigen::Vector2d &uv, int boundary, float scale, std::shared_ptr<MCVOcamera> ptr)
{
    int u = static_cast<int>(uv(0) * scale);
    int v = static_cast<int>(uv(1) * scale);

    if (u > 0 + boundary && u < static_cast<int>(float(ptr->COL) * scale) - boundary &&
        v > 0 + boundary && v < static_cast<int>(float(ptr->ROW) * scale) - boundary)
    {
        return true;
    }
    else
    {
        return false;
    }
} // function is_in_image end

// TODO: for =a multi-cam configuration
void MCVOfrontend::show_image_with_points(std::shared_ptr<MCVOcamera> ptr, size_t num_level)
{
#if SHOW_LOG_DEBUG
    LOG(INFO)
        << "Show image with points";
#endif
    int tracker_idx = ptr->tracker_idx;
    cv::Mat &img = laservisual_alligned_imgs_[tracker_idx];
    if (pub_depthimage.getNumSubscribers() != 0)
    {
        cv::Mat img_with_points = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3);

        if (img.type() == CV_32FC1)
        {
            cvtColor(img, img_with_points, cv::COLOR_GRAY2BGR);
        }
        else
        {
            //        img_with_points = img;
            img.copyTo(img_with_points);
        }

        //    cv::namedWindow("original_rgb",cv::WINDOW_NORMAL);
        //    cv::imshow("original_rgb",img_with_points);

        const float scale = 1.0f / (1 << num_level);

        // cerr << "Point size = " << imageCloud->size() << endl;
        int n = 0;
        for (auto &iter : *(ptr->imageCloud))
        {
            n++;
            if (n % 5 != 0)
                continue;

            Eigen::Vector3d xyz_ref(iter.x, iter.y, iter.z);
            Eigen::Vector2d uv_ref;

            trackerData[tracker_idx]->m_camera->spaceToPlane(xyz_ref, uv_ref);

            const float u_ref_f = uv_ref(0);
            const float v_ref_f = uv_ref(1);
            const int u_ref_i = static_cast<int>(u_ref_f);
            const int v_ref_i = static_cast<int>(v_ref_f);

            //设置最大最远距离
            float v_min = 1.0;
            float v_max = 50.0;
            //设置距离差
            float dv = v_max - v_min;
            //取深度值
            float v = xyz_ref(2);
            //        if(v>30)
            //        cout<<"v: "<<v<<endl;
            float r = 1.0;
            float g = 1.0;
            float b = 1.0;
            if (v < v_min)
                v = v_min;
            if (v > v_max)
                v = v_max;

            if (v < v_min + 0.25 * dv)
            {
                r = 0.0;
                g = 4 * (v - v_min) / dv;
            }
            else if (v < (v_min + 0.5 * dv))
            {
                r = 0.0;
                b = 1 + 4 * (v_min + 0.25 * dv - v) / dv;
            }
            else if (v < (v_min + 0.75 * dv))
            {
                r = 4 * (v - v_min - 0.5 * dv) / dv;
                b = 0.0;
            }
            else
            {
                g = 1 + 4 * (v_min + 0.75 * dv - v) / dv;
                b = 0.0;
            }

            //       std::cout << "color: " << r << ", " << g << ", " << b << std::endl;
            //        iter->r = r;
            //        iter->g = g;
            //        iter->b = b;
            // TODO: 数据类型要一致
#if 1
            cv::circle(img_with_points,
                       cv::Point(u_ref_i, v_ref_i),
                       3.5,
                       cv::Scalar(static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255)),
                       -1);
#else
            //　--------------- Error ---------------
            cv::circle(img_with_points, cv::Point(u_ref_i, v_ref_i), 3.5, cv::Scalar(r, g, b), -1);
            //　--------------- Error ---------------
#endif
        }

        cv_bridge::CvImage bridge;
        bridge.image = img_with_points;
        bridge.encoding = "rgb8";
        sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
        imageShowPointer->header.stamp = ros::Time::now();
        pub_depthimage.publish(imageShowPointer);

        //  cv::imshow("image_with_points", img_with_points);
        //  cv::waitKey(1);
    }
}
float MCVOfrontend::pointDistance(PointType p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

void MCVOfrontend::cloud_in_image(std::shared_ptr<MCVOcamera> ptr)
{
// ROS_DEBUG("cloud in image");
#if SHOW_LOG_DEBUG
    LOG(INFO) << "Cloud in image";
#endif
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInput = ptr->syncCloud;
    int max_level_ = 2;
    float scale = 1.0f / (1 << max_level_);
    int count = 0;
    int tracker_idx = ptr->tracker_idx;
    pcl::PointCloud<pcl::PointXYZRGBA> pc;
    ptr->normalize_point.clear();
#if SHOW_LOG_DEBUG
    LOG(INFO) << "cloud size:" << cloudInput->size();
#endif
    for (const auto &iter : *cloudInput)
    {
        //将点云的xyz,转换成uv
        // LOG(INFO) << "------------------------";
        // LOG(INFO) << "Compute point pixel for pt " << count++;
        pcl::PointXYZRGBA point;

        Eigen::Vector3d P(double(iter.x), double(iter.y), double(iter.z));
        // LOG(INFO) << "P:" << std::endl
        //           << P;
        Eigen::Vector2d uv;

        trackerData[tracker_idx]->m_camera->spaceToPlane(P, uv);
        // LOG(INFO) << "3d pose:" << std::endl
        //           << P << std::endl
        //           << "2d pose:" << std::endl
        //           << "(" << uv(0) << "," << uv(1) << ")";
        // LOG(INFO) << "Check inside image";
        //判断激光点云 是否在图像平面上
        if (is_in_image(uv, Boundary, scale, ptr))
        { // && iter->z < 5.0
            // LOG(INFO) << "Inside";
            int u = static_cast<int>(uv(0));
            int v = static_cast<int>(uv(1));
            cv::Vec3b bgr;
            //在KIIT图像上选取激光点的对应的像素坐标，并得到坐标对应的像素值
            // TODO: OpenCV 的数据类型不熟悉, 需要看一下OpenCV 图像访问的方法
            // LOG(INFO) << "Fetch bgr";
            if (!laservisual_alligned_imgs_[tracker_idx].empty())
                bgr = laservisual_alligned_imgs_[tracker_idx].at<cv::Vec3b>(v, u);
            else
            {
                continue;
            }

            //将像素值附着给激光点云上，也就是让激光点云具备颜色信息
            point.x = iter.x;
            point.y = iter.y;
            point.z = iter.z;

            // TODO: 下面的操作要与OpenCV 的数据类型一致
#if 0
            point.r = static_cast<uint8_t>(bgr[2] * 255.0);
            point.g = static_cast<uint8_t>(bgr[1] * 255.0);
            point.b = static_cast<uint8_t>(bgr[0] * 255.0);
            point.a = 1.0;
#else

            // LOG(INFO) << "Fetch rgb";
            point.r = static_cast<uint8_t>(bgr[2]);
            point.g = static_cast<uint8_t>(bgr[1]);
            point.b = static_cast<uint8_t>(bgr[0]);
            point.a = 1.0;
#endif
        }
        else
        {
            //如果激光点不合法，那么就采用绿色的点云
            point.r = static_cast<uint8_t>(0.0);
            point.g = static_cast<uint8_t>(255.0);
            point.b = static_cast<uint8_t>(0.0);
            point.a = 1;
        }
        //如果激光点大于0.0 就存储进来
        if (iter.z > 0.0)
        {
            pc.push_back(point);

            pcl::PointXYZI point;
            point.intensity = iter.z;
            point.x = iter.x * 10.f / iter.z;
            point.y = iter.y * 10.f / iter.z;
            point.z = 10.f;
            // LOG(INFO) << "add to normalize points";
            ptr->normalize_point.push_back(point);
        }
        // LOG(INFO) << "------------------";
    }
    pcl::copyPointCloud(pc, *(ptr->imageCloud));
    //    LOG(INFO) << "imageCloud size: " << imageCloud->size();

    if (ptr->imageCloud->size() == 0)
        return;

    // LOG(INFO) << "Insert to kd tree";
    // ptr->lidar_mutex.lock();
    synchronizer.result_mutexes[tracker_idx]->lock();
    ptr->kdTree_->setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>(ptr->normalize_point)));
    // ptr->lidar_mutex.unlock();
    synchronizer.result_mutexes[tracker_idx]->unlock();
    show_image_with_points(ptr, 0);

    laservisual_alligned_imgs_[tracker_idx].release();
} // cloud_in_image end

#if MERGELASER
void MCVOfrontend::lidar_callback(const sensor_msgs::PointCloud2ConstPtr &laser_msg)
{
    LOG(INFO) << "lidar_callback";
    static int lidar_count = -1;
    if (++lidar_count % (3 + 1) != 0)
        return;
    LOG(INFO) << "begin";
    syncCloud.reset(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*laser_msg, *syncCloud);

    // 0. listen to transform
    static tf::TransformListener listener;
    static tf::StampedTransform transform;
    try
    {
        listener.waitForTransform(LIO_World_Frame, LIO_Laser_Frame, laser_msg->header.stamp, ros::Duration(1.0));
        listener.lookupTransform(LIO_World_Frame, LIO_Laser_Frame, laser_msg->header.stamp, transform);
        LOG(INFO) << "GET VIO frontend get lidar no tf from " << LIO_World_Frame << " to " << LIO_Laser_Frame;
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR_STREAM("VIO frontend get lidar no tf from " << LIO_World_Frame << " to " << LIO_Laser_Frame << ": " << ex.what());
        return;
    }

    double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
    xCur = transform.getOrigin().x();
    yCur = transform.getOrigin().y();
    zCur = transform.getOrigin().z();
    tf::Matrix3x3 m(transform.getRotation());
    m.getRPY(rollCur, pitchCur, yawCur);
    Eigen::Affine3f T_w_lidar = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

    if (LASER_TYPE == 64)
    {
        auto cloudSize = syncCloud->points.size();
        float verticalAngle;
        int rowIdn;
        pcl::PointCloud<pcl::PointXYZ> downsampled_cloud;
        pcl::PointXYZ thisPoint;
        for (size_t i = 0; i < cloudSize; ++i)
        {
            thisPoint.x = syncCloud->points[i].x;
            thisPoint.y = syncCloud->points[i].y;
            thisPoint.z = syncCloud->points[i].z;

            verticalAngle =
                atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % 4 == 0)
            {
                downsampled_cloud.push_back(thisPoint);
            }
        }

        *syncCloud = downsampled_cloud;
    }

    // 2. downsample new cloud (save memory)
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
    static pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(syncCloud);
    downSizeFilter.filter(*laser_cloud_in_ds);
    *syncCloud = *laser_cloud_in_ds;

    // 3. filter lidar points (only keep points in camera view)
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)syncCloud->size(); ++i)
    {
        pcl::PointXYZ p = syncCloud->points[i];
        if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
            laser_cloud_in_filter->push_back(p);
    }
    *syncCloud = *laser_cloud_in_filter;

    // 4. offset T_lidar -> T_camera
#if 0  
    // pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());  
    // Eigen::Matrix<double, 4, 4> extrinsic_cam_laser;
    // extrinsic_cam_laser = Eigen::Matrix4d::Identity();
    // extrinsic_cam_laser.block<3, 3>(0, 0) = cam_laser_R;
    // extrinsic_cam_laser.block<3, 1>(0, 3) = cam_laser_T;
    // pcl::transformPointCloud(*syncCloud, *laser_cloud_offset, extrinsic_cam_laser);
    // *syncCloud = *laser_cloud_offset;
#else
    // 4. offset T_lidar -> T_camera
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
    // TODO henryzh47: This transOffset is all 0?
    Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);
    pcl::transformPointCloud(*syncCloud, *laser_cloud_offset, transOffset);
    *syncCloud = *laser_cloud_offset;
#endif

    // 5. transform new cloud into global odom frame
    pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*syncCloud, *laser_cloud_global, T_w_lidar);

    // 6. save new cloud
    double timeScanCur = laser_msg->header.stamp.toSec();
    cloudQueue.push_back(*laser_cloud_global);
    timeQueue.push_back(timeScanCur);

    // 7. pop old cloud
    while (!timeQueue.empty())
    {
        if (timeScanCur - timeQueue.front() > 5.0)
        {
            cloudQueue.pop_front();
            timeQueue.pop_front();
        }
        else
        {
            break;
        }
    }

    // 8. fuse global cloud
    depthCloud->clear();
    for (int i = 0; i < (int)cloudQueue.size(); ++i)
        *depthCloud += cloudQueue[i];

    // 9. downsample global cloud
    pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(depthCloud);
    downSizeFilter.filter(*depthCloudDS);
    *depthCloud = *depthCloudDS;

    if (pub_lidar_map.getNumSubscribers() != 0)
        publishCloud(&pub_lidar_map, depthCloud, laser_msg->header.stamp, VINS_World_Frame);
}
#else
// /velodyne_points Freq: ~10Hz
// /image_raw Freq: ~ 25Hz
// map /velodyne_points to current /image_raw ?
void MCVOfrontend::processLidar(const sensor_msgs::PointCloud2ConstPtr &laser_msg)
{
    // 同步的点云
    if (!init_pub)
    {
        init_pub = true;
        for (auto i : sensors)
        {
            if (i->type == sensor_type::MONOCULAR ||
                i->type == sensor_type::RGBD ||
                i->type == sensor_type::THERMAL)
            {
                init_pub &= dynamic_pointer_cast<MCVOcamera>(i)->init_pub;
            }
        }
        return;
    }
// for (int i = 0; i < trackerData.size(); i++)
// {
//     if (laservisual_alligned_imgs_[i].empty())
//     {
//         LOG(INFO) << "Empty laser aligned image of tracker "<<i;
//         continue;
//     }
// ROS_DEBUG("lidar_callback");
#if SHOW_LOG_DEBUG
    LOG(INFO) << "Process lidar";
#endif
    syncCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*laser_msg, *syncCloud);
    // ROS_DEBUG("Down sampling");
    if (LASER_TYPE == 64)
    {
        auto cloudSize = syncCloud->points.size();
        float verticalAngle;
        int rowIdn;
        pcl::PointCloud<pcl::PointXYZ> downsampled_cloud;
        pcl::PointXYZ thisPoint;
        for (size_t i = 0; i < cloudSize; ++i)
        {
            thisPoint.x = syncCloud->points[i].x;
            thisPoint.y = syncCloud->points[i].y;
            thisPoint.z = syncCloud->points[i].z;

            verticalAngle =
                atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % 4 == 0)
            {
                downsampled_cloud.push_back(thisPoint);
            }
        }
        *syncCloud = downsampled_cloud;
    }
    // ROS_DEBUG("Transformation");
    for (auto i : sensors)
    {
        if (i->type == sensor_type::MONOCULAR ||
            i->type == sensor_type::RGBD ||
            i->type == sensor_type::THERMAL)
        {

            Eigen::Matrix<double, 4, 4> extrinsic_cam_laser;
            std::shared_ptr<MCVOcamera> ptr_ = dynamic_pointer_cast<MCVOcamera>(i);
            extrinsic_cam_laser = Eigen::Matrix4d::Identity();
            extrinsic_cam_laser.block<3, 3>(0, 0) = ptr_->cam_laser_R;
            extrinsic_cam_laser.block<3, 1>(0, 3) = ptr_->cam_laser_T;

            //   std::cout<<"matrix:"<<extrinsic_cam_laser<<std::endl;
            // 将激光数据转化到相机坐标系下
            ptr_->syncCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*syncCloud, *ptr_->syncCloud, extrinsic_cam_laser);
            // ROS_DEBUG("Cloud in image");
            // lidar_aligned_image_mutex.lock();
            cloud_in_image(ptr_);
        }
    }
    // }
    // lidar_aligned_image_mutex.unlock();
    // ROS_DEBUG("Finish lidar callback");
}
#endif
#if MERGELASER
bool MCVOfrontend::create_sphere_cloud_kdtree(const ros::Time &stamp_cur, cv::Mat cur_image, const pcl::PointCloud<PointType>::Ptr &depth_cloud_temp)
{

    // 0.3 look up transform at current image time
    static tf::TransformListener listener2;
    static tf::StampedTransform T_w_laser;
    try
    {
        // henryzh47: IMU_Frame is actually Lidar frame in LIO
        listener2.waitForTransform(LIO_World_Frame, LIO_Laser_Frame, stamp_cur, ros::Duration(0.01));
        listener2.lookupTransform(LIO_World_Frame, LIO_Laser_Frame, stamp_cur, T_w_laser);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR_STREAM("lidar no tf: " << ex.what());
        return false;
    }

    double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
    xCur = T_w_laser.getOrigin().x();
    yCur = T_w_laser.getOrigin().y();
    zCur = T_w_laser.getOrigin().z();
    tf::Matrix3x3 m(T_w_laser.getRotation());
    m.getRPY(rollCur, pitchCur, yawCur);
    Eigen::Affine3f T_w_l = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);
    Eigen::Affine3d T_w_l_ = T_w_l.cast<double>();
    Transformd T_w_laser_(T_w_l_.rotation(), T_w_l_.translation());

    // 0.2  check if depthCloud available
    if (depth_cloud_temp->size() == 0)
        return false;

    // 0.2 transform cloud from global frame to camera frame
    pcl::PointCloud<PointType>::Ptr depth_cloud_local(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*depth_cloud_temp, *depth_cloud_local, T_w_laser_.inverse().transform());

    // 0.3. project depth cloud on a range image, filter points satcked in the same region
    float bin_res = 180.0 / (float)num_bins; // currently only cover the space in front of lidar (-90 ~ 90)
    cv::Mat rangeImage = cv::Mat(num_bins, num_bins, CV_32F, cv::Scalar::all(FLT_MAX));

    ROS_INFO("depth_cloud_local size %d ", depth_cloud_local->size());

    for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
    {
        PointType p = depth_cloud_local->points[i];
        // TODO henryzh47: This assumes the rotation between camera and lidar
        // filter points not in camera view
        if (p.x < 0 || abs(p.y / p.x) > 10 || abs(p.z / p.x) > 10)
            continue;
        // find row id in range image
        float row_angle = atan2(p.z, sqrt(p.x * p.x + p.y * p.y)) * 180.0 / M_PI + 90.0; // degrees, bottom -> up, 0 -> 360
        int row_id = round(row_angle / bin_res);
        // find column id in range image
        float col_angle = atan2(p.x, p.y) * 180.0 / M_PI; // degrees, left -> right, 0 -> 360
        int col_id = round(col_angle / bin_res);
        // id may be out of boundary
        if (row_id < 0 || row_id >= num_bins || col_id < 0 || col_id >= num_bins)
            continue;
        // only keep points that's closer
        float dist = pointDistance(p);
        if (dist < rangeImage.at<float>(row_id, col_id))
        {
            rangeImage.at<float>(row_id, col_id) = dist;
            pointsArray[row_id][col_id] = p;
        }
    }

    // 0.4. extract downsampled depth cloud from range image
    pcl::PointCloud<PointType>::Ptr depth_cloud_local_filter2(new pcl::PointCloud<PointType>());
    for (int i = 0; i < num_bins; ++i)
    {
        for (int j = 0; j < num_bins; ++j)
        {
            if (rangeImage.at<float>(i, j) != FLT_MAX)
                depth_cloud_local_filter2->push_back(pointsArray[i][j]);
        }
    }

    *depth_cloud_local = *depth_cloud_local_filter2;

    if (pub_depth_cloud.getNumSubscribers() != 0)
        publishCloud(&pub_depth_cloud, depth_cloud_local, stamp_cur, Laser_Frame);

    // 5. project depth cloud onto a unit sphere
    pcl::PointCloud<pcl::PointXYZI>::Ptr depth_cloud_unit_sphere_(new pcl::PointCloud<pcl::PointXYZI>());
    for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
    {
        pcl::PointXYZI point;
        PointType p;
        p = depth_cloud_local->points[i];
        float range = pointDistance(p);

        point.x = p.x / range;
        point.y = p.y / range;
        point.z = p.z / range;
        point.intensity = range;
        depth_cloud_unit_sphere_->push_back(point);
    }

    *depth_cloud_unit_sphere = *depth_cloud_unit_sphere_;

    if (depth_cloud_unit_sphere->size() < 10)
        return false;

    // 6. create a kd-tree using the spherical depth cloud
    kdTree_->setInputCloud(depth_cloud_unit_sphere);

    project_cloud_image(stamp_cur, cur_image, depth_cloud_local);

    return true;
}

void MCVOfrontend::get_depth(const geometry_msgs::Point32 &features_2d, pcl::PointXYZI &p)
{

    // 1. normalize 2d feature to a unit sphere
    Eigen::Vector3f feature_cur(features_2d.x, features_2d.y, features_2d.z); // z always equal to 1
    feature_cur.normalize();

    // convert to ROS standard
    p.x = feature_cur(2);
    p.y = -feature_cur(0);
    p.z = -feature_cur(1);
    p.intensity = -1;

    // 2. find the feature depth using kd-tree
    vector<int> pointSearchInd;
    vector<float> pointSearchSqDis;
    float bin_res = 180.0 / (float)num_bins; // currently only cover the space in front of lidar (-90 ~ 90)
    float dist_sq_threshold = pow(sin(bin_res / 180.0 * M_PI) * 5.0, 2);

    kdTree_->nearestKSearch(p, 3, pointSearchInd, pointSearchSqDis);

    if (pointSearchInd.size() == 3 && pointSearchSqDis[2] < dist_sq_threshold)
    {
        float r1 = depth_cloud_unit_sphere->points[pointSearchInd[0]].intensity;
        Eigen::Vector3f A(depth_cloud_unit_sphere->points[pointSearchInd[0]].x * r1,
                          depth_cloud_unit_sphere->points[pointSearchInd[0]].y * r1,
                          depth_cloud_unit_sphere->points[pointSearchInd[0]].z * r1);

        float r2 = depth_cloud_unit_sphere->points[pointSearchInd[1]].intensity;
        Eigen::Vector3f B(depth_cloud_unit_sphere->points[pointSearchInd[1]].x * r2,
                          depth_cloud_unit_sphere->points[pointSearchInd[1]].y * r2,
                          depth_cloud_unit_sphere->points[pointSearchInd[1]].z * r2);

        float r3 = depth_cloud_unit_sphere->points[pointSearchInd[2]].intensity;
        Eigen::Vector3f C(depth_cloud_unit_sphere->points[pointSearchInd[2]].x * r3,
                          depth_cloud_unit_sphere->points[pointSearchInd[2]].y * r3,
                          depth_cloud_unit_sphere->points[pointSearchInd[2]].z * r3);

        // https://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
        // demo RGB-D
        Eigen::Vector3f V(p.x,
                          p.y,
                          p.z);

        Eigen::Vector3f N = (A - B).cross(B - C);
        float s = (N(0) * A(0) + N(1) * A(1) + N(2) * A(2)) / (N(0) * V(0) + N(1) * V(1) + N(2) * V(2));

        float min_depth = min(r1, min(r2, r3));
        float max_depth = max(r1, max(r2, r3));
        if (max_depth - min_depth > 2 || s <= 0.5)
        {
            //  p.intensity=0;
            return;
        }
        else if (s - max_depth > 0)
        {
            s = max_depth;
        }
        else if (s - min_depth < 0)
        {
            s = min_depth;
        }
        // convert feature into cartesian space if depth is available
        p.x *= s;
        p.y *= s;
        p.z *= s;
        // the obtained depth here is for unit sphere, VINS estimator needs depth for normalized feature (by value z), (lidar x = camera z)
        p.intensity = p.x;
    }
}

void MCVOfrontend::project_cloud_image(const ros::Time &stamp_cur, cv::Mat imageCur, const pcl::PointCloud<PointType>::Ptr &depth_cloud_local)
{
    // visualization project points on image for visualization (it's slow!)
    if (pub_depthimage.getNumSubscribers() != 0)
    {

        vector<cv::Point2f> points_2d;
        vector<float> points_distance;

        for (int i = 0; i < (int)depth_cloud_local->size(); ++i)
        {
            // convert points from 3D to 2D
            Eigen::Vector3d p_3d(-depth_cloud_local->points[i].y,
                                 -depth_cloud_local->points[i].z,
                                 depth_cloud_local->points[i].x);
            Eigen::Vector2d p_2d;
            trackerData[0].m_camera->spaceToPlane(p_3d, p_2d);
            points_2d.push_back(cv::Point2f(p_2d(0), p_2d(1)));
            points_distance.push_back(pointDistance(depth_cloud_local->points[i]));
        }

        cv::Mat showImage, circleImage;
        showImage = imageCur.clone();
        // cv::cvtColor(imageCur, showImage, cv::COLOR_GRAY2RGB);
        circleImage = showImage.clone();

        for (int i = 0; i < (int)points_2d.size(); ++i)
        {
            float r, g, b;
            getColor(points_distance[i], 50.0, r, g, b);
            cv::circle(circleImage, points_2d[i], 0, cv::Scalar(r, g, b), 5);
        }

        cv::addWeighted(showImage, 1.0, circleImage, 0.7, 0, showImage); // blend camera image and circle image

        cv_bridge::CvImage bridge;
        bridge.image = showImage;
        bridge.encoding = "rgb8";
        sensor_msgs::Image::Ptr imageShowPointer = bridge.toImageMsg();
        imageShowPointer->header.stamp = stamp_cur;

        pub_depthimage.publish(imageShowPointer);
    }
}
#endif