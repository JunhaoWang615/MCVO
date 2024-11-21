/**
 * @file    SuperGlue.cpp
 *
 * @author  btran
 *
 */

#include <torch/script.h>
#include <torch/torch.h>

#include "SuperGlue.hpp"
#include "Utility.hpp"

namespace _cv
{
class SuperGlueImpl : public SuperGlue
{
 public:
    explicit SuperGlueImpl(const SuperGlue::Param& param);

    void match(cv::InputArray _queryDescriptors, const std::vector<cv::KeyPoint>& queryKeypoints,
               const cv::Size& querySize, cv::InputArray _trainDescriptors,
               const std::vector<cv::KeyPoint>& trainKeypoints, const cv::Size& trainSize,
               CV_OUT std::vector<cv::DMatch>& matches) const final;

 private:
    SuperGlue::Param m_param;
    torch::Device m_device;
    mutable torch::jit::script::Module m_module;
};

cv::Ptr<SuperGlue> SuperGlue::create(const Param& param)
{
    return cv::makePtr<SuperGlueImpl>(param);
}

SuperGlueImpl::SuperGlueImpl(const SuperGlue::Param& param)
    : m_param(param)
    , m_device(torch::kCPU)
{
    if (m_param.pathToWeights.empty()) {
        throw std::runtime_error("empty path to weights");
    }
    try {
        m_module = torch::jit::load(m_param.pathToWeights);
    } catch (const std::exception& e) {
        INFO_LOG("%s", e.what());
        exit(1);
    }

#if ENABLE_GPU
    if (!torch::cuda::is_available() && m_param.gpuIdx >= 0) {
        DEBUG_LOG("torch does not recognize cuda device so fall back to cpu...");
        m_param.gpuIdx = -1;
    }
#else
    DEBUG_LOG("gpu option is not enabled...");
    m_param.gpuIdx = -1;
#endif
    std::cout << "SuperGlue gpuIdx " << m_param.gpuIdx << " prepared!" << std::endl;
    if (m_param.gpuIdx >= 0) {
        torch::NoGradGuard no_grad;
        m_device = torch::Device(torch::kCUDA, m_param.gpuIdx);
    }
    m_module.eval();
    m_module.to(m_device);

    
}

void SuperGlueImpl::match(cv::InputArray _queryDescriptors, const std::vector<cv::KeyPoint>& queryKeypoints,
                          const cv::Size& querySize, cv::InputArray _trainDescriptors,
                          const std::vector<cv::KeyPoint>& trainKeypoints, const cv::Size& trainSize,
                          CV_OUT std::vector<cv::DMatch>& matches) const
{

    torch::Dict<std::string, torch::Tensor> data;
    data.insert(
        "image0_shape",
        torch::from_blob(
            std::vector<float>{1, 1, static_cast<float>(querySize.height), static_cast<float>(querySize.width)}.data(),
            {4}, torch::kFloat)
            .clone());
    data.insert(
        "image1_shape",
        torch::from_blob(
            std::vector<float>{1, 1, static_cast<float>(trainSize.height), static_cast<float>(trainSize.width)}.data(),
            {4}, torch::kFloat)
            .clone());
    data.insert("match_threshold",
                torch::from_blob(std::vector<float>{m_param.matchThreshold}.data(), {1}, torch::kFloat).clone());


    int numQueryKeyPoints = queryKeypoints.size();
    int numTrainKeyPoints = trainKeypoints.size();

    std::vector<torch::Tensor> descriptorsList = {
        torch::from_blob(_queryDescriptors.getMat().ptr<float>(),
                         {1, numQueryKeyPoints, _queryDescriptors.getMat().cols}, torch::kFloat),
        torch::from_blob(_trainDescriptors.getMat().ptr<float>(),
                         {1, numTrainKeyPoints, _trainDescriptors.getMat().cols}, torch::kFloat)};

    for (int i = 0; i < 2; ++i) {
        descriptorsList[i] = descriptorsList[i].permute({0, 2, 1}).contiguous();
        data.insert("descriptors" + std::to_string(i), std::move(descriptorsList[i]).to(m_device));
    }

    std::vector<torch::Tensor> keyPointsList = {torch::zeros({1, numQueryKeyPoints, 2}),
                                                torch::zeros({1, numTrainKeyPoints, 2})};
    std::vector<torch::Tensor> scoresList = {torch::zeros({1, numQueryKeyPoints}),
                                             torch::zeros({1, numTrainKeyPoints})};

    for (int i = 0; i < numQueryKeyPoints; ++i) {
        keyPointsList[0][0][i][0] = queryKeypoints[i].pt.y;
        keyPointsList[0][0][i][1] = queryKeypoints[i].pt.x;
        scoresList[0][0][i] = queryKeypoints[i].response;
    }

    for (int i = 0; i < numTrainKeyPoints; ++i) {
        keyPointsList[1][0][i][0] = trainKeypoints[i].pt.y;
        keyPointsList[1][0][i][1] = trainKeypoints[i].pt.x;
        scoresList[1][0][i] = trainKeypoints[i].response;
    }

    for (int i = 0; i < 2; ++i) {
        data.insert("keypoints" + std::to_string(i), std::move(keyPointsList[i]).to(m_device));
        data.insert("scores" + std::to_string(i), std::move(scoresList[i]).to(m_device));
    }

    torch::Tensor matches0;
    {
        auto outputs =
            c10::impl::toTypedDict<std::string, torch::Tensor>(m_module.forward({std::move(data)}).toGenericDict());
        matches0 = outputs.at("matches0");
        matches0 = matches0.detach().cpu();
    }

    for (int i = 0; i < numQueryKeyPoints; ++i) {
        if (matches0[0][i].item<std::int64_t>() < 0) {
            continue;
        }
        cv::DMatch match;
        match.imgIdx = 0;
        match.queryIdx = i;
        match.trainIdx = matches0[0][i].item<std::int64_t>();

        matches.emplace_back(match);
    }
}
}  // namespace _cv
