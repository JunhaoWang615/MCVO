/**
 * @file    SuperPoint.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace _cv
{
class CV_EXPORTS_W SuperPoint : public cv::Feature2D
{
 public:
    struct Param {
        // reduce input shapes can increase speed and reduce (GPU) memory consumption
        // at the cost of accuracy
        int imageHeight = 480;
        int imageWidth = 640;

        std::string pathToWeights = "";
        int borderRemove = 4;
        float confidenceThresh = 0.015;
        int distThresh = 2;  // nms. set value <= 0 to deactivate nms
        int gpuIdx = -1;     // use gpu >= 0 to specify cuda device
    };

    CV_WRAP static cv::Ptr<SuperPoint> create(const Param& param);
};
}  // namespace _cv
