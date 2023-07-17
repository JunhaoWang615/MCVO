/**
 * @file    SuperGlue.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace _cv
{
class SuperGlue
{
 public:
    struct Param {
        std::string pathToWeights = "";
        float matchThreshold = 0.1;
        int gpuIdx = -1;  // use gpu >= 0 to specify cuda device
    };

    static cv::Ptr<SuperGlue> create(const Param& param);

    virtual void match(cv::InputArray _queryDescriptors, const std::vector<cv::KeyPoint>& queryKeypoints,
                       const cv::Size& querySize, cv::InputArray _trainDescriptors,
                       const std::vector<cv::KeyPoint>& trainKeypoints, const cv::Size& trainSize,
                       CV_OUT std::vector<cv::DMatch>& matches) const = 0;
};
}  // namespace _cv
