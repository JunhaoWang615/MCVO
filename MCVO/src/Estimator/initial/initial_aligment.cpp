#include "initial_alignment.h"

/**
 * @brief   陀螺仪偏置校正
 * @optional    根据视觉SFM的结果来校正陀螺仪Bias -> Paper V-B-1
 *              主要是将相邻帧之间SFM求解出来的旋转矩阵与IMU预积分的旋转量对齐
 *              注意得到了新的Bias后对应的预积分需要repropagate
 * @param[in]   all_image_frame所有图像帧构成的map,图像帧保存了位姿、预积分量和关于角点的信息
 * @param[out]  Bgs 陀螺仪偏置
 * @return      void
 */

void solveGyroscopeBias(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, int base_cam)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        for (int c = 0; c < NUM_OF_CAM; c++)
        {
            MatrixXd tmp_A(3, 3);
            tmp_A.setZero();
            VectorXd tmp_b(3);
            tmp_b.setZero();
            Eigen::Quaterniond q_ij(frame_i->second.Twi[base_cam].rot.inverse() * frame_j->second.Twi[base_cam].rot);
            tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
            tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
            A += tmp_A.transpose() * tmp_A;
            b += tmp_A.transpose() * tmp_b;
        }
    }
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;
    // 计算出bias 重新更新预积分
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x, int base_cam)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    // VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_j;
    for (int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) =
                frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) =
                frame_i->second.Twi[base_cam].rotationMatrix().transpose() * (frame_j->second.Twi[base_cam].pos - frame_i->second.Twi[base_cam].pos) / 100.0;
            tmp_b.block<3, 1>(0, 0) =
                frame_j->second.pre_integration->delta_p + frame_i->second.Twi[base_cam].rotationMatrix().transpose() * frame_j->second.Twi[base_cam].rotationMatrix() * TIC[base_cam] - TIC[base_cam] - frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) =
                frame_i->second.Twi[base_cam].rotationMatrix().transpose() * frame_j->second.Twi[base_cam].rotationMatrix();
            tmp_A.block<3, 2>(3, 6) =
                frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * Matrix3d::Identity() * g0;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            // MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        VectorXd dg = x.segment<2>(n_state - 3);
        g0 = (g0 + lxly * dg).normalized() * G.norm();
        // double s = x(n_state - 1);
    }
    g = g0;
}

/**
 * @brief   重力矢量细化
 * @optional    重力细化，在其切线空间上用两个变量重新参数化重力 -> Paper V-B-3
                g^ = ||g|| * (g^-) + w1b1 + w2b2
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、二自由度重力参数w[w1,w2]^T、尺度s
 * @return      void
*/

void RefineGravityWithDepth(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x, int base_cam)
{
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    // VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_j;
    for (int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 8);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) =
                frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(0, 0) =
                frame_j->second.pre_integration->delta_p + frame_i->second.Twi[base_cam].rotationMatrix().transpose() * frame_j->second.Twi[base_cam].rotationMatrix() * TIC[base_cam] - TIC[base_cam] - frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * dt / 2 * g0 - frame_i->second.Twi[base_cam].rotationMatrix().transpose() * (frame_j->second.Twi[base_cam].pos - frame_i->second.Twi[base_cam].pos);

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) =
                frame_i->second.Twi[base_cam].rotationMatrix().transpose() * frame_j->second.Twi[base_cam].rotationMatrix();
            tmp_A.block<3, 2>(3, 6) =
                frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * Matrix3d::Identity() * g0;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            // MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<2, 2>() += r_A.bottomRightCorner<2, 2>();
            b.tail<2>() += r_b.tail<2>();

            A.block<6, 2>(i * 3, n_state - 2) += r_A.topRightCorner<6, 2>();
            A.block<2, 6>(n_state - 2, i * 3) += r_A.bottomLeftCorner<2, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        VectorXd dg = x.segment<2>(n_state - 2);
        g0 = (g0 + lxly * dg).normalized() * G.norm();
        // TODO：少优化尺度s
        // double s = x(n_state - 1);
    }
    g = g0;
}

bool LinearAlignment(Eigen::aligned_map<double, ImageFrame> &all_image_frame,
                     Vector3d &g, VectorXd &x, int base_cam)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) =
            frame_i->second.Twi[base_cam].rotationMatrix().transpose() * (frame_j->second.Twi[base_cam].pos - frame_i->second.Twi[base_cam].pos) / 100.0;
        tmp_b.block<3, 1>(0, 0) =
            frame_j->second.pre_integration->delta_p + frame_i->second.Twi[base_cam].rotationMatrix().transpose() * frame_j->second.Twi[base_cam].rotationMatrix() * TIC[base_cam] - TIC[base_cam];
        // cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) =
            frame_i->second.Twi[base_cam].rotationMatrix().transpose() * frame_j->second.Twi[base_cam].rotationMatrix();
        tmp_A.block<3, 3>(3, 6) = frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        // cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        // MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    LOG(INFO) << "estimated scale:" << s;
    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    LOG(INFO) << " result g     " << g.norm() << " " << g.transpose();
    if (fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x, base_cam);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    LOG(INFO) << " refine     " << g.norm() << " " << g.transpose();
    if (s < 0.0)
        return false;
    else
        return true;
}
/**
 * @brief   计算尺度，重力加速度和速度
 * @optional    速度、重力向量和尺度初始化Paper -> V-B-2
 *              相邻帧之间的位置和速度与IMU预积分出来的位置和速度对齐，求解最小二乘
 *              重力细化 -> Paper V-B-3
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、重力g、尺度s
 * @return      void
 */

bool LinearAlignmentWithDepth(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x, int base_cam)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3; // no scale now

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        // TODO：本来是6*10的矩阵，现在不用优化scale 信息
        MatrixXd tmp_A(6, 9); // no scale now
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_b.block<3, 1>(0, 0) =
            frame_j->second.pre_integration->delta_p + frame_i->second.Twi[base_cam].rotationMatrix().transpose() * frame_j->second.Twi[base_cam].rotationMatrix() * TIC[base_cam] - TIC[base_cam] - frame_i->second.Twi[base_cam].rotationMatrix().transpose() * (frame_j->second.Twi[base_cam].pos - frame_i->second.Twi[base_cam].pos);
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) =
            frame_i->second.Twi[base_cam].rotationMatrix().transpose() * frame_j->second.Twi[base_cam].rotationMatrix();
        tmp_A.block<3, 3>(3, 6) = frame_i->second.Twi[base_cam].rotationMatrix().transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();

        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
        b.tail<3>() += r_b.tail<3>();

        A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
        A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);

    g = x.segment<3>(n_state - 3);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if (fabs(g.norm() - G.norm()) > 1.0)
    {
        return false;
    }

    RefineGravityWithDepth(all_image_frame, g, x, base_cam);

    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());

    return true;
}

bool VisualIMUAlignment(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x, int base_cam)
{
    solveGyroscopeBias(all_image_frame, Bgs, base_cam); // 计算陀螺仪偏置
#if USE_DEPTH_INITIAL
    if (LinearAlignmentWithDepth(all_image_frame, g, x, base_cam)) // 计算尺度，重力加速度和速度
#else
    if (LinearAlignment(all_image_frame, g, x, base_cam)) // 计算尺度，重力加速度和速度
#endif
        return true;
    else
        return false;
}

bool VisualIMUAlignmentWithDepth(Eigen::aligned_map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x, int base_cam)
{
    solveGyroscopeBias(all_image_frame, Bgs, base_cam);            // 计算陀螺仪偏置
    if (LinearAlignmentWithDepth(all_image_frame, g, x, base_cam)) // 计算尺度，重力加速度和速度
        return true;
    else
        return false;
}

bool MultiCameraAlignment(Eigen::aligned_map<double, ImageFrame> &all_image_frame, int l, VectorXd &x)
{
    LOG(INFO) << "Into multi camera alignment. Num of cam: " << NUM_OF_CAM;
    int all_frame_count = all_image_frame.size();
    int n_state = (all_frame_count - 1) * 3;

    MatrixXd A{n_state, 2};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    Eigen::aligned_map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        if (i == l)
            continue;
        MatrixXd tmp_A(3, 2);
        VectorXd tmp_b(3);
        tmp_A.setZero();
        tmp_b.setZero();
        // frame_it->second.Twi[c] = Transformd(R_pnp, T_pnp);
        tmp_A.block<3, 1>(0, 0) = frame_i->second.Twi[0].pos;
        tmp_A.block<3, 1>(0, 1) = -frame_i->second.Twi[1].pos;
        tmp_b = frame_i->second.Twi[0].rotationMatrix() * RIC[0].transpose() * TIC[0] -
                frame_i->second.Twi[1].rotationMatrix() * RIC[1].transpose() * TIC[1];

        if (i < l)
        {
            A.block<3, 2>(i * 3, 0) = tmp_A;
            b.segment<3>(i * 3) = tmp_b;
        }
        else
        {
            A.block<3, 2>((i - 1) * 3, 0) = tmp_A;
            b.segment<3>((i - 1) * 3) = tmp_b;
        }
    }

    MatrixXd r_A = A.transpose() * A;
    VectorXd r_b = A.transpose() * b;

    r_A = r_A * 1000.0;
    r_b = r_b * 1000.0;
    x = r_A.ldlt().solve(r_b);
    for (int i = 0; i < 2; i++)
    {
        LOG(INFO) << "Scale for cam " << i << ": " << x(i);
        if (x(i) < 0)
        {
            return false;
        }
    }
    return true;
}

// For multiple cameras
// bool MultiCameraAlignment(Eigen::aligned_map<double, ImageFrame> &all_image_frame, int l, VectorXd &x, vector<bool> &state)
// {
//     LOG(INFO) << "Into multi camera alignment 2";
//     int all_frame_count = all_image_frame.size();
//     // int n_state = (all_frame_count - 1) * 3;
//     int camera_count = 0;
//     vector<int> camera_block_index;
//     for (auto i : state)
//     {
//         if (i)
//         {
//             camera_block_index.push_back(camera_count);
//         }
//         camera_count++;
//     }
//     int valid_count = (int)camera_block_index.size();

//     LOG(INFO) << valid_count;
//     if (valid_count < 2)
//     {
//         LOG(INFO) << "Less than 2 cameras are valid for alignment!";
//         return false;
//     }
//     MatrixXd A{valid_count, valid_count};

//     A.setZero();

//     VectorXd b{valid_count, 1};
//     b.setZero();
//     Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
//     int i = 0;
//     for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++, i++)
//     {
//         if (i == l)
//             continue;
//         // construct A_t
//         MatrixXd A_t(valid_count, valid_count);
//         MatrixXd b_t(valid_count, 1);
//         A_t.setZero();
//         b_t.setZero();

//         for (int c1 = 0; c1 < valid_count; c1++)
//         {
//             int base_camera = camera_block_index[c1];

//             for (int c2 = c1 + 1; c2 < valid_count; c2++)
//             {
//                 int ref_camera = camera_block_index[c2];
//                 // constructing F
//                 MatrixXd F(3, valid_count);
//                 F.block<3, 1>(0, c1) = frame_i->second.Twi[base_camera].pos;
//                 F.block<3, 1>(0, c2) = -frame_i->second.Twi[ref_camera].pos;
//                 LOG(INFO) <<"F: \n"<<F;
//                 A_t += F.transpose() * F;
//                 LOG(INFO) <<"A_t: \n"<<A_t;
//                 Vector3d theta;
//                 theta.setZero();
//                 theta = frame_i->second.Twi[ref_camera].rotationMatrix() * RIC[ref_camera].transpose() * TIC[ref_camera] -
//                         frame_i->second.Twi[base_camera].rotationMatrix() * RIC[base_camera].transpose() * TIC[base_camera];
//                 LOG(INFO) <<"theta: \n"<<theta;
//                 b_t += F.transpose() * theta;
//                 LOG(INFO) <<"b_t: \n"<<b_t;
//             }
//         }
//         A += A_t;
//         b += b_t;
//     }
//     LOG(INFO) <<"Slove";
//     LOG(INFO) <<"A:\n"<<A;
//     LOG(INFO) <<"b:\n" <<b;
//     // Eigenvalue checker
//     A = A;
//     b = b;
//     EigenSolver<MatrixXd> s(A);
//     auto eivals = s.eigenvalues();
//     LOG(INFO) << "The eigen value of A is: " << eivals;
//     double det = 1;
//     int rows = eivals.rows();
//     for (int r = 0; r < rows; r++)
//     {
//         LOG(INFO) << "Eigen value " << r << " : " << eivals(r);
//         det *= eivals(r).real();
//     }
//     if (fabs(det) > 1e-9)
//     {
//         x = A.inverse() * b;
//         LOG(INFO) << "Scales: \n"
//                   << x;
//         for (int c = 0; c < valid_count; c++)
//         {
//             if (x(c) < 0)
//                 return false;
//         }
//         return true;
//     }
//     else
//     {
//         return false;
//     }
// }

// For multiple cameras
bool MultiCameraAlignment(Eigen::aligned_map<double, ImageFrame> &all_image_frame, int l, VectorXd &x, vector<bool> &state)
{
    LOG(INFO) << "Into multi camera alignment 2";
    int all_frame_count = all_image_frame.size();
    // int n_state = (all_frame_count - 1) * 3;
    int camera_count = 0;
    vector<int> camera_block_index;
    for (auto i : state)
    {
        if (i)
        {
            camera_block_index.push_back(camera_count);
        }
        camera_count++;
    }
    int valid_count = (int)camera_block_index.size();

    LOG(INFO) << "Valid count:" << valid_count;
    if (valid_count < 2)
    {
        LOG(INFO) << "Less than 2 cameras are valid for alignment!";
        return false;
    }
    MatrixXd A{valid_count, valid_count};

    A.setZero();

    VectorXd b{valid_count, 1};
    b.setZero();

    ceres::Problem problem;

    double scales[valid_count][1];

    for (int i = 0; i < valid_count; i++)
    {
        scales[i][0] = 10.0;
        problem.AddParameterBlock(scales[i], 1);
    }

    Eigen::aligned_map<double, ImageFrame>::iterator frame_i;
    int i = 0;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++, i++)
    {
        if (i == l)
            continue;
        for (int c1 = 0; c1 < valid_count; c1++)
        {
            int base_camera = camera_block_index[c1];

            for (int c2 = c1 + 1; c2 < valid_count; c2++)
            {
                int ref_camera = camera_block_index[c2];
                LOG(INFO) << "i: " << i << " base " << base_camera << " refer " << ref_camera;
                Eigen::Matrix3d r_i = frame_i->second.Twi[base_camera].rotationMatrix();
                Eigen::Matrix3d r_j = frame_i->second.Twi[ref_camera].rotationMatrix();
                AlignmentFactor *f = new AlignmentFactor(r_i, frame_i->second.Twi[base_camera].pos,
                                                         r_j, frame_i->second.Twi[ref_camera].pos,
                                                         RIC[base_camera], TIC[base_camera],
                                                         RIC[ref_camera], TIC[ref_camera]);

                problem.AddResidualBlock(f, nullptr, scales[base_camera], scales[ref_camera]);
            }
        }
    }
    LOG(INFO) << "Solve";

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
        cout << "multi cam alignment converge" << endl;
    }
    else
    {
        LOG(INFO) << " multi cam alignment diverges!";
        return false;
    }

    VectorXd ss(valid_count);
    for (int i = 0; i < valid_count; i++)
    {
        LOG(INFO) << "Camera " << camera_block_index[i] << " scale: " << scales[i][0] / 1e4;
        if (scales[i][0] < 0)
        {
            LOG(INFO) << "Multi alignment fails";
            return false;
        }
        ss[i] = scales[i][0] /1e4;

    }
    std::cout << ss[0] << std::endl;

    x = ss;
        std::cout << "11111111111111111111111111111" << std::endl;
    return true;
}