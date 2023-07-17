/**
 * @file    SuperPoint.cpp
 *
 * @author  btran
 *
 */

#include <memory>

#include <torch/script.h>
#include <torch/torch.h>

#include "SuperPoint.hpp"
#include "Utility.hpp"

namespace
{
cv::Mat copyRows(const cv::Mat& src, const std::vector<int>& indices);
}  // namespace

namespace _cv
{
class SuperPointImpl : public SuperPoint
{
 public:
    explicit SuperPointImpl(const SuperPoint::Param& param);

    void detectAndCompute(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keyPoints,
                          cv::OutputArray _descriptors, bool useProvidedKeypoints) CV_OVERRIDE;

    int descriptorSize() const CV_OVERRIDE
    {
        return 256;
    }

    int descriptorType() const CV_OVERRIDE
    {
        return CV_32F;
    }

 private:
    SuperPoint::Param m_param;
    torch::Device m_device;
    torch::jit::script::Module m_module;
};

cv::Ptr<SuperPoint> SuperPoint::create(const Param& param)
{
    return cv::makePtr<SuperPointImpl>(param);
}

SuperPointImpl::SuperPointImpl(const SuperPoint::Param& param)
    : m_param(param)
    , m_device(torch::kCPU)
{
    if (m_param.imageHeight <= 0 || m_param.imageWidth <= 0) {
        throw std::runtime_error("dimension must be more than 0");
    }

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

    std::cout << "SuperPoint gpuIdx " << m_param.gpuIdx << " prepared!" << std::endl;
    if (m_param.gpuIdx >= 0) {
        torch::NoGradGuard no_grad;
        m_device = torch::Device(torch::kCUDA, m_param.gpuIdx);
    }
    DEBUG_LOG("use device: %s", m_device.str().c_str());
    m_module.eval();

    if (!m_device.is_cpu()) {
        m_module.to(m_device);
    }
}

void SuperPointImpl::detectAndCompute(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& keyPoints,
                                      cv::OutputArray _descriptors, bool useProvidedKeypoints)
{
    cv::Mat image = _image.getMat();
    cv::Mat mask = _mask.getMat();

    if (image.empty() || image.depth() != CV_8U) {
        CV_Error(cv::Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");
    }

    if (!mask.empty() && mask.type() != CV_8UC1) {
        CV_Error(cv::Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)");
    }

    torch::Dict<std::string, std::vector<torch::Tensor>> outputs;
    {
        cv::Mat buffer;
        cv::resize(image, buffer, cv::Size(m_param.imageWidth, m_param.imageHeight), 0, 0, cv::INTER_CUBIC);
        buffer.convertTo(buffer, CV_32FC1, 1 / 255.);

        auto x = torch::from_blob(buffer.ptr<float>(), {1, 1, m_param.imageHeight, m_param.imageWidth}, torch::kFloat);
        x = x.set_requires_grad(false);

        if (!m_device.is_cpu()) {
            x = x.to(m_device);
        }
        torch::Dict<std::string, torch::Tensor> data;
        data.insert("image", std::move(x));
        data.insert("keypoint_threshold",
                    torch::from_blob(std::vector<float>{m_param.confidenceThresh}.data(), {1}, torch::kFloat).clone());
        data.insert(
            "remove_borders",
            torch::from_blob(std::vector<std::int64_t>{m_param.borderRemove}.data(), {1}, torch::kInt64).clone());
        if (m_param.distThresh > 0) {
            data.insert(
                "nms_radius",
                torch::from_blob(std::vector<std::int64_t>{m_param.distThresh}.data(), {1}, torch::kInt64).clone());
        }

        outputs =
            c10::impl::toTypedDict<std::string, std::vector<torch::Tensor>>(m_module.forward({data}).toGenericDict());
    }

    // preparing mask
    if (!mask.empty()) {
        cv::resize(mask, mask, cv::Size(m_param.imageWidth, m_param.imageHeight), 0, 0, cv::INTER_NEAREST);
    }

    {
        keyPoints.clear();
        auto keyPointsT = std::move(outputs.at("keypoints")[0]);  // y, x
        auto scoresT = std::move(outputs.at("scores")[0]);
        auto descriptorsT = std::move(outputs.at("descriptors")[0]);  // 256 x num_keypoints
        descriptorsT = descriptorsT.permute({1, 0}).contiguous();

        if (!m_device.is_cpu()) {
            keyPointsT = keyPointsT.detach().cpu();
            scoresT = scoresT.detach().cpu();
            descriptorsT = descriptorsT.detach().cpu();
        }

        int numKeyPoints = keyPointsT.sizes()[0];
        cv::Mat descriptors = cv::Mat(cv::Size(256, numKeyPoints), CV_32F);
        std::memcpy(descriptors.ptr<float>(), descriptorsT.data_ptr<float>(), sizeof(float) * descriptorsT.numel());

        std::vector<int> keepIndices;
        keepIndices.reserve(numKeyPoints);
        for (int i = 0; i < numKeyPoints; ++i) {
            int y = keyPointsT[i][1].item<float>();
            int x = keyPointsT[i][0].item<float>();
            if (!mask.empty() && mask.ptr<uchar>(y)[x] == 0) {
                continue;
            }
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt.x = x * static_cast<float>(image.cols) / m_param.imageWidth;
            newKeyPoint.pt.y = y * static_cast<float>(image.rows) / m_param.imageHeight;
            newKeyPoint.response = scoresT[i].item<float>();
            keyPoints.emplace_back(std::move(newKeyPoint));
            keepIndices.emplace_back(i);
        }

        _descriptors.create(cv::Size(256, keepIndices.size()), CV_32F);
        std::memcpy(_descriptors.getMat().ptr<float>(), ::copyRows(descriptors, keepIndices).clone().ptr<float>(),
                    sizeof(float) * keepIndices.size() * 256);


    }
}
}  // namespace _cv

namespace
{
cv::Mat copyRows(const cv::Mat& src, const std::vector<int>& indices)
{
    int newNumRows = indices.size();
    cv::Mat dst(0, newNumRows, src.type());
    for (int i = 0; i < newNumRows; ++i) {
        dst.push_back(src.row(indices[i]));
    }
    return dst;
}
}  // namespace
