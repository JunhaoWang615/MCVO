#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../../utility/utility.h"
#include "../../utility/tic_toc.h"
#include "../parameters.h"

class ProjectionMCFactor: public ceres::SizedCostFunction<2, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProjectionMCFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);

    virtual bool
    Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    void check(double **parameters);
public:

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};
