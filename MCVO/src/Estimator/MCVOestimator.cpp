#include "MCVOestimator.h"
#include "../Utils/EigenTypes.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

MCVOEstimator::MCVOEstimator()
{
}

MCVOEstimator::MCVOEstimator(MCVOfrontend *frontend)
    : f_manager{Rs, frontend}
{
    failureCount = -1;
    LOG(INFO) << "estimator init begins";
    // 向 feature_manager 传递frame 管理器的指针
    f_manager.int_frameid2_time_frameid_ = &int_frameid2_time_frameid;
    f_manager.time_frameid2_int_frameid_ = &time_frameid2_int_frameid;
    f_manager.local_active_frames_ = &local_active_frames;

    frontend_ = frontend;
    clearState();
}

void MCVOEstimator::init(MCVOfrontend *frontend)
{
    clearState();
    f_manager = FeatureManager(Rs, frontend);
    failureCount = -1;
    LOG(INFO) << "estimator init begins";
    // 向 feature_manager 传递frame 管理器的指针
    f_manager.int_frameid2_time_frameid_ = &int_frameid2_time_frameid;
    f_manager.time_frameid2_int_frameid_ = &time_frameid2_int_frameid;
    f_manager.local_active_frames_ = &local_active_frames;

    frontend_ = frontend;
}
void MCVOEstimator::setParameter()
{
    LOG(INFO) << "Estimator set parameter";
    cam_state.resize(NUM_OF_CAM);
    tic = (Vector3d *)malloc(sizeof(Vector3d) * NUM_OF_CAM);
    ric = (Matrix3d *)malloc(sizeof(Matrix3d) * NUM_OF_CAM);
    para_Ex_Pose = (double **)malloc(sizeof(double *) * NUM_OF_CAM);
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        para_Ex_Pose[i] = (double *)malloc(sizeof(double) * SIZE_POSE);
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void MCVOEstimator::clearState()
{
    LOG(INFO) << "Clear state";
    ++failureCount;

    local_active_frames.clear();
    int_frameid2_time_frameid.clear();
    time_frameid2_int_frameid.clear();
    LOG(INFO) << "Clear Ps, Vs, As";
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    // for (int i = 0; i < NUM_OF_CAM; i++)
    // {
    //     tic[i] = Vector3d::Zero();
    //     ric[i] = Matrix3d::Identity();
    // }
    LOG(INFO) << "Clear local window frame";
    for (auto &it : localWindowFrames)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }
    solver_flag = SolverFlag::INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;

    initial_timestamp = 0;
    localWindowFrames.clear();
    td = TD;

    if (tmp_pre_integration)
        delete tmp_pre_integration;
#ifdef VISUAL_IMU_SUM_PRIOR
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;
#else
    if (last_visual_marginalization_info)
        delete last_visual_marginalization_info;
    if (last_imu_marginalization_info)
        delete last_imu_marginalization_info;
#endif

    tmp_pre_integration = nullptr;

#ifdef VISUAL_IMU_SUM_PRIOR
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();
#else
    last_visual_marginalization_info = nullptr;
    last_visual_marginalization_parameter_blocks.clear();

    last_imu_marginalization_info = nullptr;
    last_imu_marginalization_parameter_blocks.clear();
#endif

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();

    // henryzh47: VIO initialization stat for LIO
    initial_stat = 0;
    init_delay_frame_countdown = VIO_INIT_DELAY;
}

void MCVOEstimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        // if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// featureID-> [<camID, xyz, uv, velocity, depth>]
void MCVOEstimator::processImageAndLidar(const SyncCameraProcessingResults &input)
{
#if SHOW_LOG_DEBUG
    ROS_DEBUG("new image coming ------------------------------------------");
#endif
    // ROS_DEBUG("Adding feature points %lu", input.feature.size());
    // FeaturePerFrame
    // FeaturePerId
    // feature
    local_active_frames.insert(input.sync_timestamp);
    recomputeFrameId(); // 梳理局部滑窗中frame的id

    //   LOG(INFO) << "local_active_frames.size: " << local_active_frames.size();
    //    if (f_manager.addFeatureCheckParallax(frame_count, input.feature, td))

    if (f_manager.addFeatureCheckParallax(frame_count, input.sync_timestamp, input, td))
        marginalization_flag = MarginalizationFlag::MARGIN_OLD;
    else
        marginalization_flag = MarginalizationFlag::MARGIN_SECOND_NEW;
// std::string marg = (marginalization_flag == MarginalizationFlag::MARGIN_OLD ? "reject" : "accept");
#if SHOW_LOG_DEBUG
    printf("----\n File: \"%s\" \n line: %d \n function <%s>\n Content: Marginalization %s \n=====", __FILE__, __LINE__, __func__, (marginalization_flag == MarginalizationFlag::MARGIN_OLD ? "reject" : "accept"));
    for (int c = 0; c < NUM_OF_CAM; c++)
        LOG(INFO) << "number of feature in cam " << c << f_manager.getFeatureCount(c);
#endif
    Headers[frame_count].stamp = ros::Time().fromSec(input.sync_timestamp);

    //    {
    //
    //        int count = 0;
    //        std::string output;
    //        for (const auto tid : Headers) {
    //            output += std::to_string(count) + " ->" + std::to_string(tid.stamp.toSec()) + "\n";
    //            count++;
    //        }
    //
    //        LOG(INFO) << "Original WINDOW Frame: \n" << output;
    //    }
    // step 创建新的一帧数据
    ImageFrame imageframe(input);
    imageframe.pre_integration = tmp_pre_integration;
    localWindowFrames.insert(make_pair(input.sync_timestamp, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // check header stamp
#if SHOW_LOG_DEBUG
    int header_count = 0;
    for (auto i : localWindowFrames)
    {
        LOG(INFO) << fixed << "Local window frame " << header_count << " :" << i.first;
        if (header_count <= WINDOW_SIZE + 1)
            LOG(INFO) << fixed << "Time in header: " << Headers[header_count++].stamp.toSec();
        else
            LOG(INFO) << "Exceed maximum size!";
    }
#endif
    if (solver_flag == SolverFlag::INITIAL)
    {
        if (!PnP_simple_init)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if (ESTIMATE_EXTRINSIC != 2 && (input.sync_timestamp - initial_timestamp) > 0.1)
                {
                    for (int c = 0; c < NUM_OF_CAM; c++)
                    {
                        cam_state[c] = true;
                    }
                    f_manager.setCamState(cam_state);
// result = initialStructure();

#if 0
                    result = initialMultiStructure2();
#else
                    result = initialMultiStructure();
#endif
                    initial_timestamp = input.sync_timestamp;
                }
                // if init sfm success
                if (result)
                {
                    solver_flag = SolverFlag::NON_LINEAR;
                    // LOG(INFO) << "solveOdometry";
                    solveOdometry();
                    // LOG(INFO) << "Slide window";
                    slideWindow();
                    // LOG(INFO) << "Remove failure";
                    f_manager.removeFailures();
                    // LOG(INFO) << "Initialization finish!";
                    last_R = Rs[WINDOW_SIZE];
                    last_P = Ps[WINDOW_SIZE];
                    last_R0 = Rs[0];
                    last_P0 = Ps[0];
                }
                else
                {
                    for (int c = 0; c < NUM_OF_CAM; c++)
                    {
                        cam_state[c] = true;
                    }
                    f_manager.setCamState(cam_state);
                    slideWindow();
                }
            }
            else
                frame_count++;
        }
#if 0
        else
        {
            // Lidar depth support + PnP init
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            // f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count > -1)
            {
                f_manager.triangulateWithDepth(Ps, tic, ric);
                f_manager.triangulate(Ps, tic, ric);
                // LOG(INFO) << "triangulation finish";
            }
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = localWindowFrames.begin(); frame_it != localWindowFrames.end(); frame_it++)
                {
                    // frame_it->second.R = Rs[i];
                    // frame_it->second.T = Ps[i];
                    frame_it->second.Twi = Transformd(Rs[i], Ps[i]);
                    i++;
                }
                solveGyroscopeBias(localWindowFrames, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                // optimization();
                // updateLatestStates();
                solveOdometry();
                solver_flag = SolverFlag::NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
            if (frame_count < WINDOW_SIZE)
            {
                frame_count++;
                int prev_frame = frame_count - 1;
                Ps[frame_count] = Ps[prev_frame];
                Vs[frame_count] = Vs[prev_frame];
                Rs[frame_count] = Rs[prev_frame];
                Bas[frame_count] = Bas[prev_frame];
                Bgs[frame_count] = Bgs[prev_frame];
            }
        }
#endif
    }
    else
    {
        solveOdometry();
        // LOG(ERROR) << "solver costs: " << t_solve.toc() << "ms";
        //my work wang
        // est_num++;
        // if(est_num >= 30)
        // {
        //             cout << "21121212121221221111111111111111" << est_num <<endl;
        // TicToc t_solve;
        // int l;
        // Matrix3d relative_R[NUM_OF_CAM];
        // Vector3d relative_T[NUM_OF_CAM];
        // vector<bool> state;
        // VectorXd x;
        // bool result_m;
        // bool result = false;
        // // for (int i = 0; i <= NUM_OF_CAM; i++)
        // // {
        // //     state.push_back(true);
        // // }
        // result_m = mulcam_relativePose(relative_R, relative_T, l, state);
        // if(result_m)
        //     result = MultiCameraAlignment(localWindowFrames, l, x, state);
        // if(result)
        // {   
        // int base_cam = 0;
        // for (int c = 0; c < NUM_OF_CAM; c++)
        // {
        //     if (!state[c])
        //         continue;
        //     else
        //     {
        //         base_cam = c;
        //         break;
        //     }
        // }
        // for (int i = 0; i <= WINDOW_SIZE; i++)
        // {
        // Ps[i] = Ps[i]*x(base_cam);
        // Vs[i] = Vs[i]*x(base_cam);
        // Bas[i] = Bas[i]*x(base_cam);
        // Bgs[i] = Bgs[i]*x(base_cam);
        // }
        // }
        // est_num = 0;
        //
        // tic[i] = tic[i]*x(base_cam);
        //}
        if (failureDetection())
        {
            LOG(ERROR) << "failure detection!";
            failure_occur = true;
            clearState();
            setParameter();
            for (int c = 0; c < NUM_OF_CAM; c++)
            {
                cam_state[c] = true;
            }
            f_manager.setCamState(cam_state);
            LOG(ERROR) << "system reboot!";
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        //        LOG(INFO) << "marginalization costs: " << t_margin.toc() << "ms";
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }

    // henryzh47: update VIO initial stat
    if (init_delay_frame_countdown-- >= 0)
    {
        ROS_WARN("VIO initialization buffer frames");
    }
    else
    {
        initial_stat = 1;
    }

    recomputeFrameId();
} // function processImageAndLidar

void MCVOEstimator::recomputeFrameId()
{
#if SHOW_LOG_DEBUG
    LOG(INFO)
        << "Recompute frame id";
#endif
    int_frameid2_time_frameid.clear();
    time_frameid2_int_frameid.clear();

    int localwindow_id = 0;
    std::string output;
    for (const auto tid : local_active_frames)
    {
        int_frameid2_time_frameid[localwindow_id] = tid;
        time_frameid2_int_frameid[tid] = localwindow_id;
        output += std::to_string(localwindow_id) + " ->" + std::to_string(tid) + "\n";

        localwindow_id++;
    }
    // LOG(INFO) << "WINDOW Frame: \n" << output;

} // recomputeFrameId

bool MCVOEstimator::initialMultiStructure()
{
    LOG(INFO) << "Initial structure for multiple cameras";
    Eigen::aligned_vector<Eigen::aligned_vector<Quaterniond>> Q;
    Eigen::aligned_vector<Eigen::aligned_vector<Vector3d>> T;

    Eigen::aligned_vector<Eigen::aligned_map<int, Vector3d>> sfm_tracked_points;
    Eigen::aligned_vector<Eigen::aligned_vector<SFMFeature>> sfm_f;
    est_num = 0;
    Q.resize(NUM_OF_CAM);
    T.resize(NUM_OF_CAM);
    sfm_f.resize(NUM_OF_CAM);
    sfm_tracked_points.resize(NUM_OF_CAM);
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        Q[c].resize(frame_count + 1);
        T[c].resize(frame_count + 1);
#if SINGLE_CAM_DEBUG
        if (c > 0)
            continue;
#endif
        for (auto &landmark : f_manager.KeyPointLandmarks[c])
        {
            SFMFeature tmp_feature;
            tmp_feature.state = false;                   // 表示没有深度
            tmp_feature.id = landmark.second.feature_id; // landmark的id号

            for (const auto &obser_per_frame : landmark.second.obs)
            {

                auto frame_intid = time_frameid2_int_frameid.at(obser_per_frame.first);

                Vector3d pts_j = obser_per_frame.second.point;

                tmp_feature.observation.push_back(make_pair(frame_intid, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
                tmp_feature.observation_depth.emplace_back(frame_intid, obser_per_frame.second.depth_measured);
            }
            sfm_f[c].push_back(tmp_feature);
        }
    }
    int l;
    Matrix3d relative_R[NUM_OF_CAM];
    Vector3d relative_T[NUM_OF_CAM];
    vector<bool> state;
    if (!mulcam_relativePose(relative_R, relative_T, l, state))
    {
        LOG(INFO) << "Not enough features or parallax";
        return false;
    }
    else
    {
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            if (!state[c])
            {
                LOG(INFO) << "Camera " << c << " does not have enough parallex. Skip ...";
                continue;
            }
            GlobalSFM sfm;
            if (!sfm.weak_mulcam_construct(frame_count + 1, Q, T, l, c,
                                           relative_R[c], relative_T[c],
                                           sfm_f, sfm_tracked_points, ric, tic))
            {
                LOG(ERROR) << "2. global SFM fails for camera " << c;
                return false;
            }
        }
    }
    // LOG(INFO)<<"NUM OF CAM"<<NUM_OF_CAM;
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        if (!state[c])
            continue;
        // solve pnp for all frame
        Eigen::aligned_map<double, ImageFrame>::iterator frame_it;
        Eigen::aligned_map<int, Vector3d>::iterator it;

        // acceleration ratio
        double last_v, current_v, last_a, current_a;
        last_v = current_v = current_a = last_a = 0;
        double last_stamp;
        frame_it = localWindowFrames.begin();
        for (int i = 0; frame_it != localWindowFrames.end(); frame_it++)
        {
            // provide initial guess
            cv::Mat r, rvec, t, D, tmp_r;

            if ((frame_it->first) == Headers[i].stamp.toSec())
            {
                frame_it->second.is_key_frame = true;
                // First store camera pose, later trasform back to body pose
                // frame_it->second.Twi[c] = Transformd(Q[c][i].toRotationMatrix() * RIC[c].transpose(), T[c][i]);

                // nan, inf check
                Eigen::Matrix3d cam_rot = Q[c][i].toRotationMatrix();
                for (int idx = 0; idx < 9; idx++)
                {
                    if (std::isnan(cam_rot(idx)) || std::isinf(cam_rot(idx)))
                    {
                        return false;
                    }
                }
                for (int idx = 0; idx < 3; idx++)
                {
                    if (std::isnan(T[c][i](idx)) || std::isinf(T[c][i](idx)))
                    {
                        return false;
                    }
                }
                frame_it->second.Twi[c] = Transformd(Q[c][i].toRotationMatrix(), T[c][i]);
                i++;
                double traveled_distance = 0;
                if (i > 1)
                {
                    Eigen::Vec3d dist = T[c][i] - T[c][i - 1];
                    traveled_distance = dist.norm();
                }
#if 0
                if (i > 1)
                {
                    Eigen::Vec3d dist = T[c][i] - T[c][i - 1];
                    traveled_distance = dist.norm();
                    if (traveled_distance > 1e5)
                    {
                        ROS_WARN("Unexpected large travel distance, initialization fails!");
                        return false;
                    }
                    current_v = traveled_distance / (frame_it->first - last_stamp);
                    // a constraint acceleration assumption
                    if (i > 2)
                    {
                        current_a = (current_v - last_v) / (frame_it->first - last_stamp);
                        if (i > 3)
                        {
                            LOG(INFO) << "Last a " << last_a;
                            LOG(INFO) << "Current a " << current_a;
                            if (fabs(last_a) > 0.2)
                            {
                                if (fabs(current_a / last_a) > 20)
                                {

                                    ROS_WARN("Unsatisfy the constraint acceleration assumption! Initialization fails!");
                                    return false;
                                }
                            }
                            if (fabs(current_a) > 0.2)
                            {
                                if (fabs(last_a / current_a) > 20)
                                {
                                    ROS_WARN("Unsatisfy the constraint acceleration assumption! Initialization fails!");
                                    return false;
                                }
                            }
                        }
                    }
                }
#endif
                LOG(INFO) << i << " th kf:\n"
                          << Q[c][i].toRotationMatrix() << '\n'
                          << T[c][i] << "  Distance:" << traveled_distance << "-------------";

                last_v = current_v;
                last_a = current_a;
                last_stamp = frame_it->first;
                continue;
            }
            if ((frame_it->first) > Headers[i].stamp.toSec())
            {
                LOG(INFO) << "Current stamp greater than stamp headers[" << i << "]";
                i++;
            }
            LOG(INFO) << fixed << "current t: " << frame_it->first;
            LOG(INFO) << fixed << "Headers[" << i << "]: " << Headers[i].stamp.toSec();
            LOG(INFO) << frame_it->first - Headers[i].stamp.toSec();
            // Q和T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵
            // initial guess
            Matrix3d R_inital = (Q[c][i].inverse()).toRotationMatrix();
            Vector3d P_inital = -R_inital * T[c][i];
            // LOG(INFO)<<i<<"th pose:"<<std::endl<<R_inital<<std::endl<<P_inital<<std::endl<<"------------------";
            cv::eigen2cv(R_inital, tmp_r);
            cv::Rodrigues(tmp_r, rvec);
            cv::eigen2cv(P_inital, t);

            frame_it->second.is_key_frame = false;
            vector<cv::Point3f> pts_3_vector;
            vector<cv::Point2f> pts_2_vector;
            // points: map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
            for (auto &id_pts : frame_it->second.points[c])
            {
                int feature_id = id_pts.first;
                for (auto &i_p : id_pts.second)
                {
                    it = sfm_tracked_points[c].find(feature_id);
                    if (it != sfm_tracked_points[c].end())
                    {
                        Vector3d world_pts = it->second;
                        cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                        pts_3_vector.push_back(pts_3);
                        // Vector2d img_pts = i_p.second.head<2>();
                        Vector2d img_pts = i_p.head<2>();
                        cv::Point2f pts_2(img_pts(0), img_pts(1));
                        pts_2_vector.push_back(pts_2);
                    }
                }
            }
            LOG(INFO) << "pts_3_vector size : " << pts_3_vector.size();
            cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
            // cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            if (pts_3_vector.size() < 6)
            {
                cout << "pts_3_vector size " << pts_3_vector.size() << endl;
                LOG(INFO) << "Not enough points for solve pnp !";
                return false;
            }
            else
            {
                if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true))
                {
                    LOG(INFO) << "solve pnp fail!";
                    return false;
                }

                cv::Rodrigues(rvec, r);
                MatrixXd R_pnp, tmp_R_pnp;
                cv::cv2eigen(r, tmp_R_pnp);
                R_pnp = tmp_R_pnp.transpose();
                MatrixXd T_pnp;
                cv::cv2eigen(t, T_pnp);
                T_pnp = R_pnp * (-T_pnp);
                LOG(INFO) << i << " th frame (non kf):\n"
                          << tmp_R_pnp << '\n'
                          << T_pnp << "-------------";
                // frame_it->second.Twi[c] = Transformd(R_pnp * RIC[c].transpose(), T_pnp);
                frame_it->second.Twi[c] = Transformd(R_pnp, T_pnp);
            }
        }
    }
    if (multiVisualInitialAlign(l, state))
    {
        return true;
    }
    return false;
}

bool MCVOEstimator::initialMultiStructure2()
{
    LOG(INFO) << "Initial structure for 2 cameras";
    Eigen::aligned_vector<Eigen::aligned_vector<Quaterniond>> Q;
    Eigen::aligned_vector<Eigen::aligned_vector<Vector3d>> T;

    Eigen::aligned_vector<Eigen::aligned_map<int, Vector3d>> sfm_tracked_points;
    Eigen::aligned_vector<Eigen::aligned_vector<SFMFeature>> sfm_f;

    Q.resize(NUM_OF_CAM);
    T.resize(NUM_OF_CAM);
    sfm_f.resize(NUM_OF_CAM);
    sfm_tracked_points.resize(NUM_OF_CAM);
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        Q[c].resize(frame_count + 1);
        T[c].resize(frame_count + 1);
        // TODO: 使用 unordered_map 代替　list
        // TODO: 需要修改的地方
#if SINGLE_CAM_DEBUG
        if (c > 0)
            continue;
#endif
        //   LOG(INFO) << "COME HERE!";
        // LOG(INFO)<<"f_manager.KeyPointLandmarks: "<<f_manager.KeyPointLandmarks.size();
        for (auto &landmark : f_manager.KeyPointLandmarks[c])
        {
            SFMFeature tmp_feature;
            tmp_feature.state = false;                   // 表示没有深度
            tmp_feature.id = landmark.second.feature_id; // landmark的id号

            // auto observation = landmark.second.obs; unused declaration
            //       LOG(INFO)<<" landmark.second.obs.size(): "<< landmark.second.obs.size();
            for (const auto &obser_per_frame : landmark.second.obs)
            {

                auto frame_intid = time_frameid2_int_frameid.at(obser_per_frame.first);
                //   LOG(INFO)<<"frame_intid: "<<frame_intid;

                Vector3d pts_j = obser_per_frame.second.point;

                tmp_feature.observation.push_back(make_pair(frame_intid, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
                tmp_feature.observation_depth.emplace_back(frame_intid, obser_per_frame.second.depth_measured);
            }
            sfm_f[c].push_back(tmp_feature);
        }
    }
    // for (int c = 0; c < NUM_OF_CAM; c++)
    //     for (int f = 0; f < frame_count + 1; f++)
    //     {
    //         LOG(INFO) << f << " th kf:\n"
    //                   << Q[c][f].toRotationMatrix() << '\n'
    //                   << T[c][f];
    //     }
    int l;
    Matrix3d relative_R[NUM_OF_CAM];
    Vector3d relative_T[NUM_OF_CAM];
    if (!mulcam_relativePose(relative_R, relative_T, l))
    {
        LOG(INFO) << "Not enough features or parallax";
        return false;
    }
    else
    {
        for (int c = 0; c < NUM_OF_CAM; c++)
        {

            GlobalSFM sfm;
            if (!sfm.weak_mulcam_construct(frame_count + 1, Q, T, l, c,
                                           relative_R[c], relative_T[c],
                                           sfm_f, sfm_tracked_points, ric, tic))
            {
                LOG(ERROR) << "2. global SFM fails for camera " << c;
                return false;
            }
        }
    }

    // LOG(INFO)<<"NUM OF CAM"<<NUM_OF_CAM;
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        // solve pnp for all frame
        Eigen::aligned_map<double, ImageFrame>::iterator frame_it;
        Eigen::aligned_map<int, Vector3d>::iterator it;

        // acceleration ratio
        double last_v, current_v, last_a, current_a;
        last_v = current_v = current_a = last_a = 0;
        double last_stamp;
        frame_it = localWindowFrames.begin();
        for (int i = 0; frame_it != localWindowFrames.end(); frame_it++)
        {
            // provide initial guess
            cv::Mat r, rvec, t, D, tmp_r;

            if ((frame_it->first) == Headers[i].stamp.toSec())
            {
                frame_it->second.is_key_frame = true;
                // First store camera pose, later trasform back to body pose
                // frame_it->second.Twi[c] = Transformd(Q[c][i].toRotationMatrix() * RIC[c].transpose(), T[c][i]);

                // nan, inf check
                Eigen::Matrix3d cam_rot = Q[c][i].toRotationMatrix();
                for (int idx = 0; idx < 9; idx++)
                {
                    if (std::isnan(cam_rot(idx)) || std::isinf(cam_rot(idx)))
                    {
                        return false;
                    }
                }
                for (int idx = 0; idx < 3; idx++)
                {
                    if (std::isnan(T[c][i](idx)) || std::isinf(T[c][i](idx)))
                    {
                        return false;
                    }
                }
                frame_it->second.Twi[c] = Transformd(Q[c][i].toRotationMatrix(), T[c][i]);
                i++;
                double traveled_distance = 0;
                if (i > 1)
                {
                    Eigen::Vec3d dist = T[c][i] - T[c][i - 1];
                    traveled_distance = dist.norm();
                }
#if 0
                if (i > 1)
                {
                    Eigen::Vec3d dist = T[c][i] - T[c][i - 1];
                    traveled_distance = dist.norm();
                    if (traveled_distance > 1e5)
                    {
                        ROS_WARN("Unexpected large travel distance, initialization fails!");
                        return false;
                    }
                    current_v = traveled_distance / (frame_it->first - last_stamp);
                    // a constraint acceleration assumption
                    if (i > 2)
                    {
                        current_a = (current_v - last_v) / (frame_it->first - last_stamp);
                        if (i > 3)
                        {
                            LOG(INFO) << "Last a " << last_a;
                            LOG(INFO) << "Current a " << current_a;
                            if (fabs(last_a) > 0.2)
                            {
                                if (fabs(current_a / last_a) > 10)
                                {

                                    ROS_WARN("Unsatisfy the constraint acceleration assumption! Initialization fails!");
                                    return false;
                                }
                            }
                            if (fabs(current_a) > 0.2)
                            {
                                if (fabs(last_a / current_a) > 10)
                                {
                                    ROS_WARN("Unsatisfy the constraint acceleration assumption! Initialization fails!");
                                    return false;
                                }
                            }
                        }
                    }
                }
#endif
                LOG(INFO) << i << " th kf:\n"
                          << Q[c][i].toRotationMatrix() << '\n'
                          << T[c][i] << "  Distance:" << traveled_distance << "-------------";

                last_v = current_v;
                last_a = current_a;
                last_stamp = frame_it->first;
                continue;
            }
            if ((frame_it->first) > Headers[i].stamp.toSec())
            {
                LOG(INFO) << "Current stamp greater than stamp headers[" << i << "]";
                i++;
            }
            LOG(INFO) << fixed << "current t: " << frame_it->first;
            LOG(INFO) << fixed << "Headers[" << i << "]: " << Headers[i].stamp.toSec();
            LOG(INFO) << frame_it->first - Headers[i].stamp.toSec();
            // Q和T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵
            // initial guess
            Matrix3d R_inital = (Q[c][i].inverse()).toRotationMatrix();
            Vector3d P_inital = -R_inital * T[c][i];
            // LOG(INFO)<<i<<"th pose:"<<std::endl<<R_inital<<std::endl<<P_inital<<std::endl<<"------------------";
            cv::eigen2cv(R_inital, tmp_r);
            cv::Rodrigues(tmp_r, rvec);
            cv::eigen2cv(P_inital, t);

            frame_it->second.is_key_frame = false;
            vector<cv::Point3f> pts_3_vector;
            vector<cv::Point2f> pts_2_vector;
            // points: map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
            for (auto &id_pts : frame_it->second.points[c])
            {
                int feature_id = id_pts.first;
                for (auto &i_p : id_pts.second)
                {
                    it = sfm_tracked_points[c].find(feature_id);
                    if (it != sfm_tracked_points[c].end())
                    {
                        Vector3d world_pts = it->second;
                        cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                        pts_3_vector.push_back(pts_3);
                        // Vector2d img_pts = i_p.second.head<2>();
                        Vector2d img_pts = i_p.head<2>();
                        cv::Point2f pts_2(img_pts(0), img_pts(1));
                        pts_2_vector.push_back(pts_2);
                    }
                }
            }
            LOG(INFO) << "pts_3_vector size : " << pts_3_vector.size();
            cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
            // cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            if (pts_3_vector.size() < 6)
            {
                cout << "pts_3_vector size " << pts_3_vector.size() << endl;
                LOG(INFO) << "Not enough points for solve pnp !";
                return false;
            }
            else
            {
                if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true))
                {
                    LOG(INFO) << "solve pnp fail!";
                    return false;
                }

                cv::Rodrigues(rvec, r);
                MatrixXd R_pnp, tmp_R_pnp;
                cv::cv2eigen(r, tmp_R_pnp);
                R_pnp = tmp_R_pnp.transpose();
                MatrixXd T_pnp;
                cv::cv2eigen(t, T_pnp);
                T_pnp = R_pnp * (-T_pnp);
                LOG(INFO) << i << " th frame (non kf):\n"
                          << tmp_R_pnp << '\n'
                          << T_pnp << "-------------";
                // frame_it->second.Twi[c] = Transformd(R_pnp * RIC[c].transpose(), T_pnp);
                frame_it->second.Twi[c] = Transformd(R_pnp, T_pnp);
            }
        }
    }
    if (multiVisualInitialAlign(l))
    {
        return true;
    }
    return false;
}

bool MCVOEstimator::multiVisualInitialAlign(int l)
{
    LOG(INFO) << "Multi Visual align";
    TicToc t_g;
    VectorXd x;
    bool result;
    // solve scale
    if (NUM_OF_CAM > 1)
        result = MultiCameraAlignment(localWindowFrames, l, x);

    else
    {
        result = true;
        x = VectorXd(1);
        x(0) = 1;
    }

    if (!result)
    {
        ROS_ERROR("Multi-camera alignment fails!");
        return false;
    }

    // change state
    LOG(INFO) << "After alignment:";

    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = localWindowFrames[Headers[i].stamp.toSec()].Twi[0].rotationMatrix();
        Vector3d Pi = localWindowFrames[Headers[i].stamp.toSec()].Twi[0].pos * x(0);

        Ps[i] = Pi;
        Rs[i] = Ri;
        localWindowFrames[Headers[i].stamp.toSec()].is_key_frame = true;
    }
    for (int i = frame_count; i >= 0; i--)
    {
        // true scale
        Ps[i] = Ps[i] - Rs[i] * RIC[0].transpose() * TIC[0] - (Ps[0] - Rs[i] * RIC[0].transpose() * TIC[0]);
        double traveled_distance = 0;
        if (i < frame_count)
        {
            Eigen::Vec3d dist = Ps[i + 1] - Ps[i];
            traveled_distance = dist.norm();
        }
        LOG(INFO) << std::endl
                  << Rs[i] << std::endl
                  << Ps[i] << "  Distance:" << traveled_distance << "-------------";
        LOG(INFO) << "-------------";
    }
    f_manager.resetDepth();
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        TIC_TMP[i].setZero();
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    // true scale
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]), 0);
    // recover velocity
    int kv = -1;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    for (frame_i = localWindowFrames.begin(); frame_i != localWindowFrames.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            if (kv >= frame_count - 1)
            {
                Vs[kv] = Vs[kv - 1];
            }
            else
                Vs[kv] = (Ps[kv + 1] - Ps[kv]) / (Headers[kv + 1].stamp.toSec() - Headers[kv].stamp.toSec());
        }
    }
    return true;
}

bool MCVOEstimator::multiVisualInitialAlign(int l, vector<bool> &state)
{
    LOG(INFO) << "Multi Visual align";
    TicToc t_g;
    VectorXd x;
    bool result;
    // solve scale

    result = MultiCameraAlignment(localWindowFrames, l, x, state);
    if (!result)
    {
        ROS_ERROR("Multi-camera alignment fails!");
        return false;
    }
        std::cout << "11111111111111111111111111111" << std::endl;

    // change state
    LOG(INFO) << "After alignment:";

    int base_cam = 0;
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        if (!state[c])
            continue;
        else
        {
            base_cam = c;
            break;
        }
    }

    for (int i = 0; i <= frame_count; i++)
    {

        Matrix3d Ri = localWindowFrames[Headers[i].stamp.toSec()].Twi[base_cam].rotationMatrix();
        Vector3d Pi = localWindowFrames[Headers[i].stamp.toSec()].Twi[base_cam].pos * x(base_cam);

        Ps[i] = Pi;
        Rs[i] = Ri;
        localWindowFrames[Headers[i].stamp.toSec()].is_key_frame = true;
    }
    for (int i = frame_count; i >= 0; i--)
    {
        // true scale
        Ps[i] = Ps[i] - Rs[i] * RIC[base_cam].transpose() * TIC[base_cam] - (Ps[0] - Rs[i] * RIC[base_cam].transpose() * TIC[base_cam]);
        double traveled_distance = 0;
        if (i < frame_count)
        {
            Eigen::Vec3d dist = Ps[i + 1] - Ps[i];
            traveled_distance = dist.norm();
        }
        LOG(INFO) << std::endl
                  << Rs[i] << std::endl
                  << Ps[i] << "  Distance:" << traveled_distance << "-------------";
        LOG(INFO) << "-------------";
    }
    f_manager.resetDepth();
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        TIC_TMP[i].setZero();
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    // true scale
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]), base_camera);
    // recover velocity
    int kv = -1;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    for (frame_i = localWindowFrames.begin(); frame_i != localWindowFrames.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            if (kv >= frame_count - 1)
            {
                Vs[kv] = Vs[kv - 1];
            }
            else
                Vs[kv] = (Ps[kv + 1] - Ps[kv]) / (Headers[kv + 1].stamp.toSec() - Headers[kv].stamp.toSec());
        }
    }
    LOG(INFO) << "mul cam align finish";
    return true;
}

bool MCVOEstimator::initialStructure()
{
    TicToc t_sfm;
    // check imu observibility
    {
        double sum_t = 0;
        Eigen::aligned_map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = localWindowFrames.begin(), frame_it++; frame_it != localWindowFrames.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            sum_t += dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)localWindowFrames.size() - 1);
        double var = 0;
        for (frame_it = localWindowFrames.begin(), frame_it++; frame_it != localWindowFrames.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)localWindowFrames.size() - 1));
        // ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            LOG(INFO) << "IMU excitation not enouth!";
            // return false;
        }
        LOG(INFO) << "Time length:" << sum_t;
    }
    // global sfm
    Eigen::aligned_vector<Eigen::aligned_vector<Quaterniond>> Q;
    Eigen::aligned_vector<Eigen::aligned_vector<Vector3d>> T;

    Eigen::aligned_vector<Eigen::aligned_map<int, Vector3d>> sfm_tracked_points;
    Eigen::aligned_vector<Eigen::aligned_vector<SFMFeature>> sfm_f;

    Q.resize(NUM_OF_CAM);
    T.resize(NUM_OF_CAM);
    sfm_f.resize(NUM_OF_CAM);
    sfm_tracked_points.resize(NUM_OF_CAM);
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        Q[c].resize(frame_count + 1);
        T[c].resize(frame_count + 1);
        // TODO: 使用 unordered_map 代替　list
        // TODO: 需要修改的地方
#if SINGLE_CAM_DEBUG
        if (c > 0)
            continue;
#endif
        //   LOG(INFO) << "COME HERE!";
        // LOG(INFO)<<"f_manager.KeyPointLandmarks: "<<f_manager.KeyPointLandmarks.size();
        for (auto &landmark : f_manager.KeyPointLandmarks[c])
        {
            SFMFeature tmp_feature;
            tmp_feature.state = false;                   // 表示没有深度
            tmp_feature.id = landmark.second.feature_id; // landmark的id号

            // auto observation = landmark.second.obs; unused declaration
            //       LOG(INFO)<<" landmark.second.obs.size(): "<< landmark.second.obs.size();
            for (const auto &obser_per_frame : landmark.second.obs)
            {

                auto frame_intid = time_frameid2_int_frameid.at(obser_per_frame.first);
                //   LOG(INFO)<<"frame_intid: "<<frame_intid;

                Vector3d pts_j = obser_per_frame.second.point;

                tmp_feature.observation.push_back(make_pair(frame_intid, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
                tmp_feature.observation_depth.emplace_back(frame_intid, obser_per_frame.second.depth_measured);
            }
            sfm_f[c].push_back(tmp_feature);
        }
    }
    Eigen::aligned_vector<Eigen::aligned_vector<SFMFeature>> sfm_f_depth = sfm_f;
// LOG(INFO) << "Already Pass!"<<"num of sfm_f: "<<sfm_f.size();
// assert(frame_count==1000);
#if SHOW_LOG_DEBUG
    LOG(INFO) << "Relative pose";
#endif
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    int base_cam;
    bool init_without_depth = true;
    // with scale recovery
#if 1
    if (depth_support_init)
    {
        init_without_depth = false;
        {
#if 1
            if (!relativePosePnP(relative_R, relative_T, l, base_cam))
#else
            if (!relativePoseICP(relative_R, relative_T, l, base_cam))
#endif
            {
                LOG(INFO) << "1. Initialization using depth fails! Try default method";
                init_without_depth = true;
            }
            else
            {
                GlobalSFM sfm;
                if (!sfm.construct(frame_count + 1, Q, T, l, base_cam,
                                   relative_R, relative_T,
                                   sfm_f_depth, sfm_tracked_points, ric, tic))
                {
                    LOG(ERROR) << "2. global SFM using depth fails! Try default method";
                    sfm_tracked_points.clear();
                    init_without_depth = true;
                }
                else
                {
                    // solve pnp for all frame
                    // TODO: 为啥又要PNP一次？
                    // 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
                    Eigen::aligned_map<double, ImageFrame>::iterator frame_it;
                    Eigen::aligned_map<int, Vector3d>::iterator it;
                    frame_it = localWindowFrames.begin();
                    for (int i = 0; frame_it != localWindowFrames.end(); frame_it++)
                    {
                        // provide initial guess
                        cv::Mat r, rvec, t, D, tmp_r;
                        if ((frame_it->first) == Headers[i].stamp.toSec())
                        {
                            frame_it->second.is_key_frame = true;
                            for (int c = 0; c < NUM_OF_CAM; c++)
                                frame_it->second.Twi[c] = Transformd(Q[c][i].toRotationMatrix() * RIC[c].transpose(), T[c][i]);
                            i++;
                            double traveled_distance = 0;
                            if (i > 1)
                            {
                                Eigen::Vec3d dist = T[base_cam][i] - T[base_cam][i - 1];
                                traveled_distance = dist.norm();
                            }
                            LOG(INFO) << i << " th kf:\n"
                                      << Q[base_cam][i].toRotationMatrix() << '\n'
                                      << T[base_cam][i] << "  Distance:" << traveled_distance << "-------------";
                            continue;
                        }
                        if ((frame_it->first) > Headers[i].stamp.toSec())
                        {
                            i++;
                        }
                        for (int c = 0; c < NUM_OF_CAM; c++)
                        {
#if SINGLE_CAM_DEBUG
                            if (c > 0)
                                continue;
#endif
                            // Q和T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵
                            // initial guess
                            Matrix3d R_inital = (Q[c][i].inverse()).toRotationMatrix();
                            Vector3d P_inital = -R_inital * T[c][i];
                            // LOG(INFO)<<i<<"th pose:"<<std::endl<<R_inital<<std::endl<<P_inital<<std::endl<<"------------------";
                            cv::eigen2cv(R_inital, tmp_r);
                            cv::Rodrigues(tmp_r, rvec);
                            cv::eigen2cv(P_inital, t);

                            frame_it->second.is_key_frame = false;
                            vector<cv::Point3f> pts_3_vector;
                            vector<cv::Point2f> pts_2_vector;
                            // points: map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
                            for (auto &id_pts : frame_it->second.points[c])
                            {
                                int feature_id = id_pts.first;
                                for (auto &i_p : id_pts.second)
                                {
                                    it = sfm_tracked_points[c].find(feature_id);
                                    if (it != sfm_tracked_points[c].end())
                                    {
                                        Vector3d world_pts = it->second;
                                        cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                                        pts_3_vector.push_back(pts_3);
                                        // Vector2d img_pts = i_p.second.head<2>();
                                        Vector2d img_pts = i_p.head<2>();
                                        cv::Point2f pts_2(img_pts(0), img_pts(1));
                                        pts_2_vector.push_back(pts_2);
                                    }
                                }
                            }
                            cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
                            // cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
                            if (pts_3_vector.size() < 6)
                            {
                                cout << "pts_3_vector size " << pts_3_vector.size() << endl;
                                LOG(INFO) << "Not enough points for solve pnp ! Try default method";
                                init_without_depth = true;
                                break;
                            }
                            else
                            {
                                /**
                                 *bool cv::solvePnP(    求解pnp问题
                                 *   InputArray  objectPoints,   特征点的3D坐标数组
                                 *   InputArray  imagePoints,    特征点对应的图像坐标
                                 *   InputArray  cameraMatrix,   相机内参矩阵
                                 *   InputArray  distCoeffs,     失真系数的输入向量
                                 *   OutputArray     rvec,       旋转向量
                                 *   OutputArray     tvec,       平移向量
                                 *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
                                 *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
                                 *)
                                 */

                                if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true))
                                {
                                    ROS_DEBUG("solve pnp fail! Try default method");
                                    init_without_depth = true;
                                    break;
                                }

                                cv::Rodrigues(rvec, r);
                                MatrixXd R_pnp, tmp_R_pnp;
                                cv::cv2eigen(r, tmp_R_pnp);
                                // 这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
                                R_pnp = tmp_R_pnp.transpose();
                                MatrixXd T_pnp;
                                cv::cv2eigen(t, T_pnp);
                                T_pnp = R_pnp * (-T_pnp);
                                LOG(INFO) << i << " th frame (non kf):\n"
                                          << tmp_R_pnp << '\n'
                                          << T_pnp << "-------------";
                                frame_it->second.Twi[c] = Transformd(R_pnp * RIC[c].transpose(), T_pnp);
                            }
                        }
                    }
                    // Rs Ps ric init
                    // 进行视觉惯性联合初始化
                    if (!init_without_depth)
                    {
                        // debug uses without depth version
                        // #if 1
                        bool success = false;
                        if (!estimate_scale)
                        {
                            success = visualInitialAlignWithDepth(base_cam);
                        }
                        else
                        {
                            success = visualInitialAlign(base_cam);
                        }
                        // if (visualInitialAlignWithDepth())
                        // #else
                        // if (visualInitialAlign())
                        // #endif
                        if (success)
                        {
                            LOG(INFO) << "Use lidar depth initialization success.";
                            return true;
                        }
                        else
                        {
                            ROS_INFO("misalign visual structure with IMU using depth! Try default method");
                            init_without_depth = true;
                        }
                    }
                }
            }
        }
    }
#endif
    // without scale recovery
    // 保证具有足够的视差,由F矩阵恢复Rt
    // 第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
    // 此处的relative_R，relative_T为当前帧到参考帧（第l帧）的坐标系变换Rt
    // relative_T has no scale
    if (init_without_depth && !skip_default_init_method)
    {
        LOG(INFO) << "Use lidar depth initialization fails. Try to use default method >>>>>>>>>";
        if (!relativePose(relative_R, relative_T, l, base_cam))
        {
            LOG(INFO) << "Not enough features or parallax; Move device around";
            return false;
        }
        LOG(INFO) << "Camera R \n"
                  << relative_R;
        LOG(INFO) << "Camera T \n"
                  << relative_T;

        // 对窗口中每个图像帧求解sfm问题
        // 得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points。
        GlobalSFM sfm;
        if (!sfm.construct(frame_count + 1, Q, T, l, base_cam,
                           relative_R, relative_T,
                           sfm_f, sfm_tracked_points, ric, tic))
        {
            LOG(ERROR) << "global SFM failed!";
            marginalization_flag = MarginalizationFlag::MARGIN_OLD;
            return false;
        }

        // solve pnp for all frame
        // TODO: 为啥又要PNP一次？
        // 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
        Eigen::aligned_map<double, ImageFrame>::iterator frame_it;
        Eigen::aligned_map<int, Vector3d>::iterator it;
        frame_it = localWindowFrames.begin();
        for (int i = 0; frame_it != localWindowFrames.end(); frame_it++)
        {
            // provide initial guess
            cv::Mat r, rvec, t, D, tmp_r;
            if ((frame_it->first) == Headers[i].stamp.toSec())
            {
                frame_it->second.is_key_frame = true;
                for (int c = 0; c < NUM_OF_CAM; c++)
                {
#if SINGLE_CAM_DEBUG
                    if (c > 0)
                        continue;
#endif
                    frame_it->second.Twi[c] = Transformd(Q[c][i].toRotationMatrix() * RIC[c].transpose(), T[c][i]);
                }
                i++;
                double traveled_distance = 0;
                if (i > 1)
                {
                    Eigen::Vec3d dist = T[base_cam][i] - T[base_cam][i - 1];
                    traveled_distance = dist.norm();
                }
                LOG(INFO) << i << " th kf:\n"
                          << Q[base_cam][i].toRotationMatrix() << '\n'
                          << T[base_cam][i] << "  Distance:" << traveled_distance << "-------------";
                continue;
            }
            if ((frame_it->first) > Headers[i].stamp.toSec())
            {
                i++;
            }
            for (int c = 0; c < NUM_OF_CAM; c++)
            {
#if SINGLE_CAM_DEBUG
                if (c > 0)
                    continue;
#endif
                // Q和T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵
                Matrix3d R_inital = (Q[c][i].inverse()).toRotationMatrix();
                Vector3d P_inital = -R_inital * T[c][i];
                cv::eigen2cv(R_inital, tmp_r);
                cv::Rodrigues(tmp_r, rvec);
                cv::eigen2cv(P_inital, t);

                frame_it->second.is_key_frame = false;
                vector<cv::Point3f> pts_3_vector;
                vector<cv::Point2f> pts_2_vector;
                // points: map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
                for (auto &id_pts : frame_it->second.points[c])
                {
                    int feature_id = id_pts.first;
                    for (auto &i_p : id_pts.second)
                    {
                        it = sfm_tracked_points[c].find(feature_id);
                        if (it != sfm_tracked_points[c].end())
                        {
                            Vector3d world_pts = it->second;
                            cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                            pts_3_vector.push_back(pts_3);
                            Vector2d img_pts = i_p.head<2>();
                            cv::Point2f pts_2(img_pts(0), img_pts(1));
                            pts_2_vector.push_back(pts_2);
                        }
                    }
                }
                cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
                if (pts_3_vector.size() < 6)
                {
                    cout << "pts_3_vector size " << pts_3_vector.size() << endl;
                    LOG(INFO) << "Not enough points for solve pnp !";
                    return false;
                }
                /**
                 *bool cv::solvePnP(    求解pnp问题
                 *   InputArray  objectPoints,   特征点的3D坐标数组
                 *   InputArray  imagePoints,    特征点对应的图像坐标
                 *   InputArray  cameraMatrix,   相机内参矩阵
                 *   InputArray  distCoeffs,     失真系数的输入向量
                 *   OutputArray     rvec,       旋转向量
                 *   OutputArray     tvec,       平移向量
                 *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
                 *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
                 *)
                 */

                if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true))
                {
                    ROS_DEBUG("solve pnp fail!");
                    return false;
                }
                cv::Rodrigues(rvec, r);
                MatrixXd R_pnp, tmp_R_pnp;
                cv::cv2eigen(r, tmp_R_pnp);
                // 这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
                R_pnp = tmp_R_pnp.transpose();
                MatrixXd T_pnp;
                cv::cv2eigen(t, T_pnp);
                T_pnp = R_pnp * (-T_pnp);
                LOG(INFO) << i << " th frame (non kf):\n"
                          << tmp_R_pnp << '\n'
                          << T_pnp << "-------------";
                frame_it->second.Twi[c] = Transformd(R_pnp * RIC[c].transpose(), T_pnp);
            }
        }
        // Rs Ps ric init
        // 进行视觉惯性联合初始化
#if 1
        if (visualInitialAlign(base_cam))
#else
        if (visualInitialAlignWithDepth(base_cam))
#endif
            return true;
        else
        {
            ROS_INFO("misalign visual structure with IMU");
            return false;
        }
    }
    return false;
} // function initialStructure

// TODO: add init for other cams
bool MCVOEstimator::visualInitialAlign(int base_cam)
{
    TicToc t_g;
    VectorXd x;
    // solve scale
    bool result = VisualIMUAlignment(localWindowFrames, Bgs, g, x, base_cam);
    if (!result)
    {
        ROS_ERROR("solve g failed!");
        return false;
    }

    // change state
    LOG(INFO) << "After alignment:";
    for (int i = 0; i <= frame_count; i++)
    {

        Matrix3d Ri = localWindowFrames[Headers[i].stamp.toSec()].Twi[base_cam].rotationMatrix();
        Vector3d Pi = localWindowFrames[Headers[i].stamp.toSec()].Twi[base_cam].pos;

        Ps[i] = Pi;
        Rs[i] = Ri;
        localWindowFrames[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    f_manager.resetDepth();

    // triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        TIC_TMP[i].setZero();
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    // Ps up-to-scale, no tic as approximation
    // triangulated depth up-to-scale for all kpts
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]), base_cam);
    // f_manager.triangulateWithDepth(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    LOG(INFO) << "Scale:" << s;
    // s = scale;
    // LOG(INFO)<<"Reset scale to "<<s;
    // ROS_DEBUG("the scale is %f\n", s);
    //  do repropagate here
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
    {
        // true scale
        Ps[i] = s * Ps[i] - Rs[i] * TIC[base_cam] - (s * Ps[0] - Rs[0] * TIC[base_cam]);
        double traveled_distance = 0;
        if (i < frame_count)
        {
            Eigen::Vec3d dist = Ps[i + 1] - Ps[i];
            traveled_distance = dist.norm();
        }
        LOG(INFO) << std::endl
                  << Rs[i] << std::endl
                  << Ps[i] << "  Distance:" << traveled_distance << "-------------";
        LOG(INFO) << "-------------";
    }
    int kv = -1;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    for (frame_i = localWindowFrames.begin(); frame_i != localWindowFrames.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.Twi[base_cam].rot * x.segment<3>(kv * 3);
        }
    }
    // all kpts depth go to true scale
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        for (auto &landmark : f_manager.KeyPointLandmarks[c])
        {
            landmark.second.used_num = landmark.second.obs.size();
            if (!(landmark.second.used_num >= 2 && time_frameid2_int_frameid.at(landmark.second.kf_id) < WINDOW_SIZE - 2))
                continue;
            landmark.second.estimated_depth *= s;
        }
    }
    // ! Modified algorithm
    f_manager.triangulate(Ps, &(tic[0]), &(ric[0]), base_cam);
    // Retriangulate to get true depth
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */

bool MCVOEstimator::visualInitialAlignWithDepth(int base_cam)
{
    TicToc t_g;
    VectorXd x;
    // solve scale

    // 计算陀螺仪偏置，尺度，重力加速度和速度
    bool result = VisualIMUAlignmentWithDepth(localWindowFrames, Bgs, g, x, base_cam);
    if (!result)
    {
        ROS_ERROR("solve g failed!");
        return false;
    }

    // change state
    // 得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = localWindowFrames[Headers[i].stamp.toSec()].Twi[base_cam].rotationMatrix();
        Vector3d Pi = localWindowFrames[Headers[i].stamp.toSec()].Twi[base_cam].pos;

        Ps[i] = Pi;
        Rs[i] = Ri;
        localWindowFrames[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // 将所有特征点的深度置为-1
    f_manager.resetDepth();

    // triangulat on cam pose , no tic
    // 重新计算特征点的深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        TIC_TMP[i].setZero();
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]), base_cam);
    // f_manager.triangulateWithDepth(Ps, &(TIC_TMP[0]), &(RIC[0]));

    // 陀螺仪的偏置bgs改变，重新计算预积分
    //  do repropagate here
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // ROS_ERROR("before %f | %f | %f\n", Ps[1].x(), Ps[1].y(), Ps[1].z());//shan add
    // 将Ps、Vs、depth 不用缩放尺度
    for (int i = frame_count; i >= 0; i--)
        // Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换
        Ps[i] = Ps[i] - Rs[i] * TIC[0] - (Ps[0] - Rs[0] * TIC[0]);
    // ROS_ERROR("after  %f | %f | %f\n", Ps[1].x(), Ps[1].y(), Ps[1].z());//shan add
    int kv = -1;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    for (frame_i = localWindowFrames.begin(); frame_i != localWindowFrames.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            // Vs为优化得到的速度
            Vs[kv] = frame_i->second.Twi[base_cam].rot * x.segment<3>(kv * 3);
        }
    }
    for (int i = frame_count; i >= 0; i--)
    {
        LOG(INFO) << i << " th kf:\n"
                  << Rs[i] << '\n'
                  << Ps[i] << '\n'
                  << "V: \n"
                  << Vs[i].norm() << '\n'
                  << "-------------";
    }
    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;

    auto first_frame_tid = int_frameid2_time_frameid.rbegin()->second;
    auto first_frame_index = int_frameid2_time_frameid.rbegin()->first;
    Transformd T_i_meas_first = localWindowFrames.at(first_frame_tid).Tw_imu_meas;
    bool laser_vio_sync = localWindowFrames.at(first_frame_tid).laser_odom_vio_sync;

#if 1
    if (laser_vio_sync == true)
    {

        Matrix3d rot_diff = T_i_meas_first.rotationMatrix() * Rs[first_frame_index].transpose();
        Vector3d tran_diff = T_i_meas_first.pos - Ps[first_frame_index];

        // 所有变量从参考坐标系c0旋转到世界坐标系w (IMU的坐标系)
        for (int i = 0; i <= frame_count; i++)
        {
            Ps[i] = rot_diff * Ps[i] + tran_diff; // shan 转到世界坐标系下？ 没明白
            Rs[i] = rot_diff * Rs[i];
            Vs[i] = rot_diff * Vs[i];

            // ROS_ERROR("%d farme's t is %f | %f | %f\n",i, Ps[i].x(), Ps[i].y(), Ps[i].z());//shan add
        }
        ROS_INFO("\033[success correct the first pose!.\033[0m");
        ROS_DEBUG_STREAM("g0     " << g.transpose());
        ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());
    }
    else
    {
        ROS_INFO("\033[initialization first pose correction has something wrong!\033[0m");
    }

#else
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    // 所有变量从参考坐标系c0旋转到世界坐标系w (IMU的坐标系)
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i]; // shan 转到世界坐标系下？ 没明白
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
        // ROS_ERROR("%d farme's t is %f | %f | %f\n",i, Ps[i].x(), Ps[i].y(), Ps[i].z());//shan add
    }
#endif

    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

bool MCVOEstimator::mulcam_relativePose(Matrix3d *relative_R, Vector3d *relative_T, int &l)
{
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Eigen::aligned_vector<Eigen::aligned_vector<pair<Vector3d, Vector3d>>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        bool flag = true;
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            if (corres[c].size() > 20 && flag == true)
            {
                ROS_INFO_STREAM("corres size between " << i << " and " << WINDOW_SIZE << ": " << corres[c].size());
                // 计算平均视差
                double sum_parallax = 0;
                double average_parallax;
                for (size_t j = 0; j < corres[c].size(); j++)
                {
                    // 第j个对应点在第i帧和最后一帧的(x,y)
                    Vector2d pts_0(corres[c][j].first(0) / corres[c][j].first(2), corres[c][j].first(1) / corres[c][j].first(2));
                    Vector2d pts_1(corres[c][j].second(0) / corres[c][j].second(2), corres[c][j].second(1) / corres[c][j].second(2));
                    double parallax = (pts_0 - pts_1).norm();
                    sum_parallax = sum_parallax + parallax;
                }
                average_parallax = 1.0 * sum_parallax / int(corres[c].size());

                ROS_INFO("average_parallax %f", average_parallax);
                LOG(INFO) << "average_parallax * 460:" << average_parallax * 460;
                // 判断是否满足初始化条件：视差>30和内点数满足要求
                // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
                Matrix3d relative_R_c;
                Vector3d relative_T_c;
                if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres[c], relative_R_c, relative_T_c))
                {
                    ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                              average_parallax * 1490,
                              l);
                    relative_R[c] = relative_R_c;
                    relative_T[c] = relative_T_c;
                }
                else
                {
                    flag = false;
                }
            }
            else
            {
                flag = false;
            }
        }
        if (flag)
        {
            l = i;
            return true;
        }
        else
        {
            continue;
        }
    }
    return false;
}

bool MCVOEstimator::mulcam_relativePose(Matrix3d *relative_R, Vector3d *relative_T, int &l, vector<bool> &state)
{
    LOG(INFO) << "Multicam relative pose";
    // search in window to maximize the number of cameras that satisfy the parallex constraint
    Eigen::aligned_vector<Eigen::aligned_vector<Matrix3d>> Rs;
    Eigen::aligned_vector<Eigen::aligned_vector<Vector3d>> Ts;
    Rs.resize(WINDOW_SIZE);
    Ts.resize(WINDOW_SIZE);
    // [count, [states]]
    vector<pair<int, vector<bool>>> valid_count;
    valid_count.resize(WINDOW_SIZE);
    for (int w = 0; w < WINDOW_SIZE; w++)
    {
        valid_count[w].first = 0;
        valid_count[w].second.resize(NUM_OF_CAM);
        Rs[w].resize(NUM_OF_CAM);
        Ts[w].resize(NUM_OF_CAM);
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            valid_count[w].second[c] = false;
        }
    }
    LOG(INFO) << "Start finding correspondance";
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        LOG(INFO) << "Window: " << i;
        Eigen::aligned_vector<Eigen::aligned_vector<pair<Vector3d, Vector3d>>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            LOG(INFO) << "Camera: " << c << " : " << corres[c].size();
            if (corres[c].size() > 20)
            {
                double sum_parallax = 0;
                double average_parallax;
                for (size_t j = 0; j < corres[c].size(); j++)
                {
                    Vector2d pts_0(corres[c][j].first(0) / corres[c][j].first(2), corres[c][j].first(1) / corres[c][j].first(2));
                    Vector2d pts_1(corres[c][j].second(0) / corres[c][j].second(2), corres[c][j].second(1) / corres[c][j].second(2));
                    double parallax = (pts_0 - pts_1).norm();
                    sum_parallax = sum_parallax + parallax;
                }
                average_parallax = 1.0 * sum_parallax / int(corres[c].size());
                LOG(INFO) << "Average Parallex * 460" << average_parallax * 460;
                if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres[c], Rs[i][c], Ts[i][c]))
                {
                    valid_count[i].first += 1;
                    valid_count[i].second[c] = true;
                    
                }
            }
        }

        LOG(INFO) << "Valid count: " << valid_count[i].first;
        //得到所有摄像头效果都很好的帧作为参考帧（下方代码
        if (valid_count[i].first == NUM_OF_CAM)
        {
            LOG(INFO) << "All cameras have good visual infomation at " << l << "Frame";
            l = i;
            state = valid_count[i].second;
            for (int c = 0; c < NUM_OF_CAM; c++)
            {
                relative_R[c] = Rs[l][c];
                relative_T[c] = Ts[l][c];
            }
            return true;
        }
    }
    //选择尽可能多的效果好的摄像头的帧作为参考帧（下方代码
    int max_count = 0;
    int max_idx = 0;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        if (valid_count[i].first > max_count)
        {
            max_idx = i;
            max_count = valid_count[i].first;
        }
    }
    l = max_idx;
    state = valid_count[max_idx].second;
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        if (state[c])
        {
            relative_R[c] = Rs[l][c];
            relative_T[c] = Ts[l][c];
        }
    }
    if (max_count < 2)
        return false;
    LOG(INFO) << "Select " << l << " Frame as reference";
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
        LOG(INFO) << "Camera " << c << " state: " << state[c];
    }
    return true;
}

/**
 * @brief   判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 * @Description    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 保存滑动窗口中与当前帧满足初始化条件的那一帧
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool MCVOEstimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l, int &base_cam)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Eigen::aligned_vector<Eigen::aligned_vector<pair<Vector3d, Vector3d>>> corres;
        // corres = f_manager.getCorresponding(i, WINDOW_SIZE);

        // TODO：深度值参与初始化过程
        // 寻找第i帧到窗口最后一帧的对应特征点
#if 0
        corres = f_manager.getCorrespondingWithDepth(i, WINDOW_SIZE);
#else
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
#endif
        int max_size = 0;
        int max_idx = 0;
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            if ((int)corres[c].size() > max_size)
            {
                max_size = corres[c].size();
                max_idx = c;
            }
        }
        if (max_size > 20)
        {
            ROS_INFO_STREAM("corres size: " << corres.size());
            // 计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (size_t j = 0; j < corres[max_idx].size(); j++)
            {
                // 第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[max_idx][j].first(0) / corres[max_idx][j].first(2), corres[max_idx][j].first(1) / corres[max_idx][j].first(2));
                Vector2d pts_1(corres[max_idx][j].second(0) / corres[max_idx][j].second(2), corres[max_idx][j].second(1) / corres[max_idx][j].second(2));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres[max_idx].size());

            ROS_INFO("average_parallax %f", average_parallax);
            LOG(INFO) << "average_parallax * 460:" << average_parallax * 460;
            // 判断是否满足初始化条件：视差>30和内点数满足要求
            // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
#if 0
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT_PNP(corres, relative_R, relative_T)) {
                //                Matrix3d relative_R2; Vector3d relative_T2;
                //                m_estimator.solveRelativeRT(corres, relative_R2, relative_T2);
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                          average_parallax * 460,
                          l);
                return true;
            }
#else
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres[max_idx], relative_R, relative_T))
            {
                //                Matrix3d relative_R2; Vector3d relative_T2;
                //                m_estimator.solveRelativeRT(corres, relative_R2, relative_T2);
                l = i;
                base_cam = max_idx;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                          average_parallax * 1490,
                          l);
                return true;
            }
#endif
            else
            {
#if SINGLE_CAM_DEBUG
                return false; // delete after debug
#endif
#if SHOW_DATA
                printf("base_cam initialization fails \n");
#endif
                for (int c = 0; c < (int)corres.size(); c++)
                {
                    if (c == max_idx)
                        continue;
                    if ((int)corres[c].size() <= 20)
                    {
#if SHOW_DATA
                        printf("cam %d does not have enough features \n", c);
#endif
                        continue;
                    }
                    sum_parallax = 0;
                    average_parallax = 0;
                    // printf("------------------------------------------------------------------------------------------\n \033[47;31mFile: \"%s\", line: %d, \033[0m \033[47;34mfunction <%s>\033[0m\n ======================================================\n", __FILE__, __LINE__, __func__);
                    for (int j = 0; j < (int)corres[c].size(); j++)
                    {
                        Vector2d pts_0(corres[c][j].first(0), corres[c][j].first(1));
                        Vector2d pts_1(corres[c][j].second(0), corres[c][j].second(1));
                        double parallax = (pts_0 - pts_1).norm();
                        sum_parallax = sum_parallax + parallax;
                    }
                    average_parallax = 1.0 * sum_parallax / int(corres[c].size());

                    if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres[c], relative_R, relative_T))
                    {
                        l = i;
                        base_cam = c;
#if SHOW_DATA
                        printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                        printf("parallax * 460 %f, choose cam %d \n", average_parallax * 460, base_cam);
#endif
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

/**
 * @brief   判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 * @Description    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 保存滑动窗口中与当前帧满足初始化条件的那一帧
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool MCVOEstimator::relativePoseICP(Matrix3d &relative_R, Vector3d &relative_T, int &l, int &base_cam)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Eigen::aligned_vector<Eigen::aligned_vector<pair<Vector3d, Vector3d>>> corres;

        // TODO：深度值参与初始化过程
        // 寻找第i帧到窗口最后一帧的对应特征点
        corres = f_manager.getCorrespondingWithDepth(i, WINDOW_SIZE);
        int max_size = 0;
        int max_idx = 0;
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            if ((int)corres[c].size() > max_size)
            {
                max_size = corres[c].size();
                max_idx = c;
            }
        }
        if (max_size > 20)
        {
            ROS_INFO_STREAM("corres size: " << corres[max_idx].size());
            // 计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (size_t j = 0; j < corres[max_idx].size(); j++)
            {
                // 第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[max_idx][j].first(0) / corres[max_idx][j].first(2), corres[max_idx][j].first(1) / corres[max_idx][j].first(2));
                Vector2d pts_1(corres[max_idx][j].second(0) / corres[max_idx][j].second(2), corres[max_idx][j].second(1) / corres[max_idx][j].second(2));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres[max_idx].size());

            ROS_INFO("average_parallax %f", average_parallax);

            // 判断是否满足初始化条件：视差>30和内点数满足要求
            // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
#if 0
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT_PNP(corres, relative_R, relative_T)) {
                //                Matrix3d relative_R2; Vector3d relative_T2;
                //                m_estimator.solveRelativeRT(corres, relative_R2, relative_T2);
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                          average_parallax * 460,
                          l);
                return true;
            }
#else
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT_ICP(corres[max_idx], relative_R, relative_T))
            {
                //                Matrix3d relative_R2; Vector3d relative_T2;
                //                m_estimator.solveRelativeRT(corres, relative_R2, relative_T2);
                l = i;
                base_cam = max_idx;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                          average_parallax * 1490,
                          l);
                return true;
            }
            else
            {
#if SINGLE_CAM_DEBUG
                return false; // delete after debug
#endif
#if SHOW_DATA
                printf("base_cam initialization fails \n");
#endif
                for (int c = 0; c < (int)corres.size(); c++)
                {
                    if (c == max_idx)
                        continue;
                    if ((int)corres[c].size() <= 20)
                    {
#if SHOW_DATA
                        printf("cam %d does not have enough features \n", c);
#endif
                        continue;
                    }
                    sum_parallax = 0;
                    average_parallax = 0;
                    // printf("------------------------------------------------------------------------------------------\n \033[47;31mFile: \"%s\", line: %d, \033[0m \033[47;34mfunction <%s>\033[0m\n ======================================================\n", __FILE__, __LINE__, __func__);
                    for (int j = 0; j < (int)corres[c].size(); j++)
                    {
                        Vector2d pts_0(corres[c][j].first(0), corres[c][j].first(1));
                        Vector2d pts_1(corres[c][j].second(0), corres[c][j].second(1));
                        double parallax = (pts_0 - pts_1).norm();
                        sum_parallax = sum_parallax + parallax;
                    }
                    average_parallax = 1.0 * sum_parallax / int(corres[c].size());

                    if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres[c], relative_R, relative_T))
                    {
                        l = i;
                        base_cam = c;
#if SHOW_DATA
                        printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                        printf("parallax * 460 %f, choose cam %d \n", average_parallax * 460, base_cam);
#endif
                        return true;
                    }
                }
            }
#endif
        }
    }
    return false;
}

/**
 * @brief   判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 * @Description    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 保存滑动窗口中与当前帧满足初始化条件的那一帧
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool MCVOEstimator::relativePosePnP(Matrix3d &relative_R, Vector3d &relative_T, int &l, int &base_cam)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Eigen::aligned_vector<Eigen::aligned_vector<pair<Vector3d, Vector3d>>> corres;

        // TODO：深度值参与初始化过程
        // 寻找第i帧到窗口最后一帧的对应特征点
        corres = f_manager.getCorrespondingWithDepth(i, WINDOW_SIZE);
        int max_size = 0;
        int max_idx = 0;
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            if ((int)corres[c].size() > max_size)
            {
                max_size = corres[c].size();
                max_idx = c;
            }
        }
        if (max_size > 20)
        {
            // ROS_INFO_STREAM("corres size: " << corres.size());
            LOG(INFO) << "corres[" << max_idx << "] size:" << corres[max_idx].size();
            // 计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (size_t j = 0; j < corres[max_idx].size(); j++)
            {
                // 第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[max_idx][j].first(0) / corres[max_idx][j].first(2), corres[max_idx][j].first(1) / corres[max_idx][j].first(2));
                Vector2d pts_1(corres[max_idx][j].second(0) / corres[max_idx][j].second(2), corres[max_idx][j].second(1) / corres[max_idx][j].second(2));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres[max_idx].size());

            ROS_INFO("average_parallax %f", average_parallax);
            LOG(INFO) << "average_parallax * 460:" << average_parallax * 460;

            // 判断是否满足初始化条件：视差>20和内点数满足要求 30 too high
            // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
            if (average_parallax * 460 > 20 && m_estimator.solveRelativeRT_PNP2(corres[max_idx], relative_R, relative_T))
            {
                //                Matrix3d relative_R2; Vector3d relative_T2;
                //                m_estimator.solveRelativeRT(corres, relative_R2, relative_T2);
                l = i;
                base_cam = max_idx;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                          average_parallax * 460,
                          l);
                return true;
            }
            else
            {
#if SINGLE_CAM_DEBUG
                return false; // delete after debug
#endif
#if SHOW_DATA
                printf("base_cam initialization fails \n");
#endif
                for (int c = 0; c < (int)corres.size(); c++)
                {
                    if (c == max_idx)
                        continue;
                    if ((int)corres[c].size() <= 20)
                    {
#if SHOW_DATA
                        printf("cam %d does not have enough features \n", c);
#endif
                        continue;
                    }
                    sum_parallax = 0;
                    average_parallax = 0;
                    // printf("------------------------------------------------------------------------------------------\n \033[47;31mFile: \"%s\", line: %d, \033[0m \033[47;34mfunction <%s>\033[0m\n ======================================================\n", __FILE__, __LINE__, __func__);
                    for (int j = 0; j < (int)corres[c].size(); j++)
                    {
                        Vector2d pts_0(corres[c][j].first(0), corres[c][j].first(1));
                        Vector2d pts_1(corres[c][j].second(0), corres[c][j].second(1));
                        double parallax = (pts_0 - pts_1).norm();
                        sum_parallax = sum_parallax + parallax;
                    }
                    average_parallax = 1.0 * sum_parallax / int(corres[c].size());

                    if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT_PNP2(corres[c], relative_R, relative_T))
                    {
                        l = i;
                        base_cam = c;
#if SHOW_DATA
                        printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                        printf("parallax * 460 %f, choose cam %d \n", average_parallax * 460, base_cam);
#endif
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void MCVOEstimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == SolverFlag::NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulateWithDepth(Ps, tic, ric, base_camera);
        f_manager.triangulate(Ps, tic, ric, base_camera);
        //        LOG(INFO) << "triangulation costs: " << t_tri.toc();
        // LOG(INFO) << "optimization";
        optimization();
        // LOG(INFO) << "Update frame pose";
        updateFramePose();
    }
}

void MCVOEstimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }

    // TODO: 3D点更新
    f_manager.depth2InvDepth();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void MCVOEstimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = false;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    // TODO: 3D 点更新
    // ------------------
    f_manager.invDepth2Depth();
    //-------------------
    if (ESTIMATE_EXTRINSIC)
    { // avoid cumulative error when transform quationd to rotation matrix

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                         .toRotationMatrix();
        }
    }
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if (relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw =
            Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        // cout << "vins relo " << endl;
        // cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        // cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = false;
    }
}

void MCVOEstimator::updateFramePose()
{
    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
    {
        auto timestamp = Headers[i].stamp.toSec();
        if (localWindowFrames.find(timestamp) != localWindowFrames.end())
        {
            for (int c = 0; c < NUM_OF_CAM; c++)
            {
                localWindowFrames[timestamp].Twi[c].rot = Rs[i];
                localWindowFrames[timestamp].Twi[c].pos = Ps[i];
            }
        }
    }
}
bool MCVOEstimator::failureDetection()
{
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
#if SINGLE_CAM_DEBUG
        if (c > 0)
            continue;
#endif
        if (f_manager.last_track_num[c] < 2)
        {
            ROS_INFO(" little feature in %d cam: %d \n", c, f_manager.last_track_num[c]);


            // return true;
        }
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_ERROR(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());

        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_ERROR(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());

        return true;
    }
    // TODO (henryzh47): change to 6m/s
    if (Vs[WINDOW_SIZE].norm() > 10.0)
    {
        ROS_ERROR("VINS big speed %f, restart estimator!", Vs[WINDOW_SIZE].norm());

        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 40) //原来为20
    {
        ROS_ERROR_STREAM(" big translation: " << (tmp_P - last_P).norm());

        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 3) //原来为1.5
    {
        ROS_ERROR_STREAM(" big z translation: " << abs(tmp_P.z() - last_P.z()));

        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_ERROR(" big delta_angle ");
        // return true;
    }
    return false;
}

void MCVOEstimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    // loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            //      LOG(INFO) << "fix extinsic param";
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            LOG(INFO) << "estimate extinsic param";
    }
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        // problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    vector2double();

#ifdef VISUAL_IMU_SUM_PRIOR
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, nullptr,
                                 last_marginalization_parameter_blocks);
    }

#else
    if (last_visual_marginalization_info)
    {
        MarginalizationFactor
            *visual_marginalization_factor = new MarginalizationFactor(last_visual_marginalization_info);
        problem.AddResidualBlock(visual_marginalization_factor, nullptr,
                                 last_visual_marginalization_parameter_blocks);
        LOG(INFO) << "[visual_marginalization_parameter_blocks] size: " << last_visual_marginalization_parameter_blocks.size();
    }

    if (last_imu_marginalization_info)
    {
        MarginalizationFactor *imu_marginalization_factor = new MarginalizationFactor(last_imu_marginalization_info);
        problem.AddResidualBlock(imu_marginalization_factor, nullptr,
                                 last_imu_marginalization_parameter_blocks);

        LOG(INFO) << "[imu_marginalization_parameter_blocks] size: " << last_imu_marginalization_parameter_blocks.size();
    }
#endif
    // remove IMU residual
    // for (int i = 0; i < WINDOW_SIZE; i++)
    // {
    //     int j = i + 1;
    //     if (pre_integrations[j]->sum_dt > 10.0)
    //         continue;
    //     IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
    //     problem.AddResidualBlock(imu_factor, nullptr, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    // }
    int f_m_cnt = 0;

    // TODO: 使用 unordered map 代替　list
    for (int c = 0; c < NUM_OF_CAM; c++)
    {
#if SINGLE_CAM_DEBUG
        if (c != base_camera)
            continue;
#endif
        for (auto &landmark : f_manager.KeyPointLandmarks[c])
        {
            landmark.second.used_num = landmark.second.obs.size();
            if (!(landmark.second.used_num >= 2 &&
                  time_frameid2_int_frameid.at(landmark.second.kf_id) < WINDOW_SIZE - 2))
                continue;

            auto host_tid = landmark.second.kf_id; // 第一个观测值对应的就是主导真
            int host_id = time_frameid2_int_frameid.at(host_tid);
            const auto &obs = landmark.second.obs;
            Vector3d pts_i = obs.at(host_tid).point;

            for (const auto &it_per_frame : obs)
            {
                auto target_tid = it_per_frame.first;
                int target_id = time_frameid2_int_frameid.at(target_tid);

                if (host_tid == target_tid)
                    continue;

                Vector3d pts_j = it_per_frame.second.point;
                if (ESTIMATE_TD)
                {
                    ProjectionTdFactor *f_td =
                        new ProjectionTdFactor(pts_i,
                                               pts_j,
                                               obs.at(host_tid).velocity,
                                               it_per_frame.second.velocity,
                                               obs.at(host_tid).cur_td,
                                               it_per_frame.second.cur_td,
                                               obs.at(host_tid).uv.y(),
                                               it_per_frame.second.uv.y());
                    problem.AddResidualBlock(f_td,
                                             loss_function,
                                             para_Pose[host_id],
                                             para_Pose[target_id],
                                             para_Ex_Pose[c],
                                             landmark.second.data.data(),
                                             para_Td[0]);

                    if (landmark.second.estimate_flag == KeyPointLandmark::EstimateFlag::DIRECT_MEASURED)
                        problem.SetParameterBlockConstant(landmark.second.data.data());
                }
                else
                {
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f,
                                             loss_function,
                                             para_Pose[host_id],
                                             para_Pose[target_id],
                                             para_Ex_Pose[c],
                                             landmark.second.data.data());

                    if (landmark.second.estimate_flag == KeyPointLandmark::EstimateFlag::DIRECT_MEASURED)
                        problem.SetParameterBlockConstant(landmark.second.data.data());
                }
                f_m_cnt++;
            }
        }
    }
    // TODO: 使用 unordered map 代替　list

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    if (relocalization_info)
    {
        if (c_cur == c_old)
        {
            // printf("set relocalization factor! \n");
            ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
            problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);

            // TODO: 使用 unordered map 代替　list
            for (auto &match_point : match_points)
            {
                int feature_id = match_point.z();

                // 查看landmarks数据库中能否找到id号相同的landmark
                if (f_manager.KeyPointLandmarks[c_cur].find(feature_id) != f_manager.KeyPointLandmarks[c_cur].end())
                {
                    auto &landmark = f_manager.KeyPointLandmarks[c_cur].at(feature_id);

                    // 确定landmark是否合法
                    landmark.used_num = landmark.obs.size();
                    if (!(landmark.used_num >= 2 and time_frameid2_int_frameid.at(landmark.kf_id) < WINDOW_SIZE - 2))
                    {
                        continue;
                    }

                    // 如果landmark合法
                    auto host_tid = landmark.kf_id;
                    int host_id = time_frameid2_int_frameid.at(host_tid);

                    Vector3d pts_i = landmark.obs.at(host_tid).point;
                    Vector3d pts_j = match_point;
                    pts_j[2] = 1.0;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);

                    // TODO: 感觉需要把逆深度优化的写入到landmark中
                    // -------------------------------------
                    problem.AddResidualBlock(f,
                                             loss_function,
                                             para_Pose[host_id],
                                             relo_Pose,
                                             para_Ex_Pose[0],
                                             landmark.data.data());
                    // ------------------------------------
                }
            }
        }
        // TODO: 使用 unordered map 代替　list
    }
    // my work wang

    // vector<int> camera_block_index;
    // for (int camera_count = 0; camera_count < NUM_OF_CAM; camera_count++;)
    // {
    //     camera_block_index.push_back(camera_count);
    // }
    // double scales[NUM_OF_CAM][1];

    // for (int i = 0; i < NUM_OF_CAM; i++)
    // {
    //     scales[i][0] = 10.0;
    //     problem.AddParameterBlock(scales[i], 1);
    // }

    // Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    // int i = 0;
    // for (frame_i = localWindowFrames.begin(); frame_i != localWindowFrames.end(); frame_i++, i++)
    // {
    //     if (i == l)
    //         continue;
    //     for (int c1 = 0; c1 < valid_count; c1++)
    //     {
    //         int base_camera = camera_block_index[c1];

    //         for (int c2 = c1 + 1; c2 < valid_count; c2++)
    //         {
    //             int ref_camera = camera_block_index[c2];
    //             LOG(INFO) << "i: " << i << " base " << base_camera << " refer " << ref_camera;
    //             Eigen::Matrix3d r_i = frame_i->second.Twi[base_camera].rotationMatrix();
    //             Eigen::Matrix3d r_j = frame_i->second.Twi[ref_camera].rotationMatrix();
    //             AlignmentFactor *f = new AlignmentFactor(r_i, frame_i->second.Twi[base_camera].pos,
    //                                                      r_j, frame_i->second.Twi[ref_camera].pos,
    //                                                      RIC[base_camera], TIC[base_camera],
    //                                                      RIC[ref_camera], TIC[ref_camera]);

    //             problem.AddResidualBlock(f, nullptr, scales[base_camera], scales[ref_camera]);
    //         }
    //     }
    // }
    // LOG(INFO) << "Solve";
    // 

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MarginalizationFlag::MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();

    TicToc t_whole_marginalization;

    if (marginalization_flag == MarginalizationFlag::MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // {
        //     if (pre_integrations[1]->sum_dt < 10.0)
        //     {
        //         IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
        //         ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, nullptr,
        //                                                                        vector<double *>{para_Pose[0],
        //                                                                                         para_SpeedBias[0],
        //                                                                                         para_Pose[1],
        //                                                                                         para_SpeedBias[1]},
        //                                                                        vector<int>{0, 1});
        //         marginalization_info->addResidualBlockInfo(residual_block_info);
        //     }
        // }
        int v_count = 0;
        {
            // int feature_index = -1;
            for (int c = 0; c < NUM_OF_CAM; c++)
            {
#if SINGLE_CAM_DEBUG
                if (c != base_camera)
                    continue;
#endif
                // TODO: 使用 unordered map 代替　list
                for (auto &landmark : f_manager.KeyPointLandmarks[c])
                {
                    landmark.second.used_num = landmark.second.obs.size();
                    if (!(landmark.second.used_num >= 2 && time_frameid2_int_frameid.at(landmark.second.kf_id) < WINDOW_SIZE - 2))
                        continue;

                    auto host_tid = landmark.second.kf_id;
                    int host_id = time_frameid2_int_frameid.at(host_tid);

                    if (host_id != 0)
                        continue;

                    const auto &obs = landmark.second.obs;

                    Vector3d pts_i = obs.at(host_tid).point;

                    for (const auto &it_per_frame : obs)
                    {

                        auto target_tid = it_per_frame.first;
                        int target_id = time_frameid2_int_frameid.at(target_tid);

                        if (host_tid == target_tid)
                            continue;

                        Vector3d pts_j = it_per_frame.second.point;

                        if (ESTIMATE_TD)
                        {
                            ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i,
                                                                              pts_j,
                                                                              obs.at(host_tid).velocity,
                                                                              it_per_frame.second.velocity,
                                                                              obs.at(host_tid).cur_td,
                                                                              it_per_frame.second.cur_td,
                                                                              obs.at(host_tid).uv.y(),
                                                                              it_per_frame.second.uv.y());

                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                           vector<double *>{
                                                                                               para_Pose[host_id],
                                                                                               para_Pose[target_id],
                                                                                               para_Ex_Pose[c],
                                                                                               landmark.second.data.data(),
                                                                                               para_Td[0]},
                                                                                           vector<int>{0, 3});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{
                                                                                               para_Pose[host_id],
                                                                                               para_Pose[target_id],
                                                                                               para_Ex_Pose[c],
                                                                                               landmark.second.data.data()},
                                                                                           vector<int>{0, 3});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                    v_count++;
                }
                
            }
        }
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        // LOG(INFO) << v_count;
        if (v_count != 0)
            marginalization_info->marginalize();
        // LOG(INFO) << "Finish marg";
        ROS_DEBUG("marginalization %f ms", t_margin.toc()); 

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
#if SINGLE_CAM_DEBUG
            if (i != base_camera)
                continue;
#endif
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        }
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {   
        LOG(INFO) << "";
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks),
                       std::end(last_marginalization_parameter_blocks),
                       para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
#if SINGLE_CAM_DEBUG
                if (i > 0)
                    continue;
#endif
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            }
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    //    LOG(INFO) << "whole marginalization costs: " << t_whole_marginalization.toc();

    //    LOG(INFO) << "whole time for ceres: " << t_whole.toc();
}

void MCVOEstimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MarginalizationFlag::MARGIN_OLD)
    {

        // TODO: 删除关键帧
        local_active_frames.erase(int_frameid2_time_frameid[0]);

        double t_0 = Headers[0].stamp.toSec();
        if (frame_count == WINDOW_SIZE)
        {

            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            Eigen::aligned_map<double, ImageFrame>::iterator it_0;
            // TODO: 这里会移动滑窗
            if (true || solver_flag == SolverFlag::INITIAL)
            {
                it_0 = localWindowFrames.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
                for (auto it = localWindowFrames.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = nullptr;
                }
                localWindowFrames.erase(localWindowFrames.begin(), it_0);
                localWindowFrames.erase(t_0);
            }

            slideWindowOld();
        }
    }
    else
    {

        //      LOG(INFO) << "marginal new second frame begin";

        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            // 删除次新帧
            local_active_frames.erase(int_frameid2_time_frameid[WINDOW_SIZE - 1]);
            // if (solver_flag == SolverFlag::INITIAL)
            localWindowFrames.erase(int_frameid2_time_frameid[WINDOW_SIZE - 1]);
            slideWindowNew();
            //         LOG(INFO) << "marginal new second frame end";
        }
    }
}

// real marginalization is removed in solve_ceres()
void MCVOEstimator::slideWindowNew()
{
    sum_of_front++;
    // 剔除次新帧的观测
    f_manager.removeOneFrameObservation(int_frameid2_time_frameid[WINDOW_SIZE - 1]);
}
// real marginalization is removed in solve_ceres()
void MCVOEstimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = (solver_flag == SolverFlag::NON_LINEAR);
    //   LOG(INFO) << "shift_depth: " << (shift_depth ? "true" : "false");
    if (shift_depth)
    {
        f_manager.removeOneFrameObservationAndShiftDepth(int_frameid2_time_frameid[0],
                                                         Ps,
                                                         tic, ric);
    }
    else
        f_manager.removeOneFrameObservation(int_frameid2_time_frameid[0]);
}

void MCVOEstimator::setReloFrame(double _frame_stamp,
                                 int _frame_index,
                                 Eigen::aligned_vector<Vector3d> &_match_points,
                                 const Vector3d &_relo_t,
                                 const Matrix3d &_relo_r,
                                 int c_cur_,
                                 int c_old_)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    c_cur = c_cur_;
    c_old = c_old_;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        if (relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = true;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}
