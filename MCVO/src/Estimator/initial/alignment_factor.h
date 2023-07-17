#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>

class AlignmentFactor : public ceres::SizedCostFunction<3, 1, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // base camera, reference camera
    AlignmentFactor(Eigen::Matrix3d r_i_, Eigen::Vector3d t_i_,
                    Eigen::Matrix3d r_j_, Eigen::Vector3d t_j_,
                    Eigen::Matrix3d ric_i_, Eigen::Vector3d tic_i_,
                    Eigen::Matrix3d ric_j_, Eigen::Vector3d tic_j_);

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Matrix3d r_i, r_j, ric_i, ric_j;
    Eigen::Vector3d t_i, t_j, tic_i, tic_j;
};