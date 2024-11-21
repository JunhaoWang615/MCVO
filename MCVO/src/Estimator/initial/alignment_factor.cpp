#include "alignment_factor.h"

#include "../../utility/utility.h"

AlignmentFactor::AlignmentFactor(Eigen::Matrix3d r_i_, Eigen::Vector3d t_i_,
                                 Eigen::Matrix3d r_j_, Eigen::Vector3d t_j_,
                                 Eigen::Matrix3d ric_i_, Eigen::Vector3d tic_i_,
                                 Eigen::Matrix3d ric_j_, Eigen::Vector3d tic_j_) : r_i(r_i_),
                                                                                   r_j(r_j_),
                                                                                   t_i(t_i_),
                                                                                   t_j(t_j_),
                                                                                   ric_i(ric_i_),
                                                                                   tic_i(tic_i_),
                                                                                   ric_j(ric_j_),
                                                                                   tic_j(tic_j_)
                                                                                   {};

bool AlignmentFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Vector2d s;
    s.setZero();
    // base_cam
    s[0] = parameters[0][0];

    LOG(INFO) << "s[0]: " << s[0];
    // ref_cam
    s[1] = parameters[1][0];

    LOG(INFO) << "s[1]: " << s[1];
    Eigen::Matrix<double, 3, 2> J;

    if (jacobians)
    {
        // Eigen::MatrixXd J{3, cam_count_};
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 1>> jacob_base_cam(jacobians[0]);
            jacob_base_cam = -r_j.transpose() * t_i;
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 1>> jacob_refe_cam(jacobians[1]);
            jacob_refe_cam = r_j.transpose() * t_j;
        }
    }
    J.block<3, 1>(0, 0) = -r_j.transpose() * t_i;
    J.block<3, 1>(0, 1) = r_j.transpose() * t_j;

    Eigen::Map<Eigen::Vector3d> residual(residuals);
    LOG(INFO) << ric_j * tic_i - ric_j.transpose() * tic_j;
    residual = (ric_j * tic_i - ric_j.transpose() * tic_j) * 1e4 + J * s;

    return true;
}