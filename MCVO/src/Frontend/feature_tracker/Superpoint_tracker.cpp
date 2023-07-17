#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include "Superpoint_tracker.h"
#include "super_include/torch_cpp.hpp"

using namespace MCVO;
using namespace torch;
using namespace nn;

// int SupFeatureTracker::n_id = 0;

bool SupFeatureTracker::inBorder(const cv::Point2f &pt)
{
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE<= img_x && img_x < cam->COL - BORDER_SIZE&& BORDER_SIZE<= img_y && img_y < cam->ROW - BORDER_SIZE;
}


void SupFeatureTracker::reduceVector(vector<cv::KeyPoint> &v, vector<uchar> &status)
{
    // cout << "vec size:" << v.size() << endl;
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
    {
        if (status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}
// void reduceVector(vector<int> &v, vector<uchar> status)
// {
//     int j = 0;
//     for (int i = 0; i < int(v.size()); i++)
//         if (status[i])
//             v[j++] = v[i];
//     v.resize(j);
// }

SupFeatureTracker::SupFeatureTracker() : TrackerBase()
{
    n_id = 0;
    // std::string usr_device = "cuda";
    // std::string model_pt = "/home/wang/catkin_ws/src/MCVO-main/MCVO/models/superpoint_v1.pt";
    // std::cout<< usr_device << std::endl;
    // std::cout<< model_pt << std::endl;

    // // choose device cuda | cpu
    // torch::Device device = torch::kCPU;
    // if (usr_device.compare("cuda") == 0) {
    //     if (torch::cuda::is_available()) {
    //         device = torch::kCUDA;
    //         std::cout << "Using CUDA" << std::endl;
    //     } else {
    //         std::cout << "CUDA is not available! Using CPU." << std::endl;
    //     }
    // } else {
    //     std::cout << "Using CPU" << std::endl;
    // }
    
    // // load model

    // model = torch::jit::load(model_pt);
    // model.to(device);
    // EPSILON = 1e-19;
    //     std::cout << "asccascasc" << std::endl;
    LOG(INFO) << "superglue().initial start";

    _cv::SuperPoint::Param superPointParam;
    superPointParam.pathToWeights = "/home/wang/catkin_ws/src/MCVO-main-new/MCVO/src/Frontend/feature_tracker/models/superpoint_model.pt";
    superPointParam.imageWidth = 350;
    superPointParam.imageHeight = 350;
    superPointParam.distThresh = 2;
    superPointParam.borderRemove = 4;
    superPointParam.confidenceThresh = 0.015;
    superPointParam.gpuIdx = 0;

    superPoint = _cv::SuperPoint::create(superPointParam);


    _cv::SuperPoint::Param superPointParam_pin;
    superPointParam_pin.pathToWeights = "/home/wang/catkin_ws/src/MCVO-main-new/MCVO/src/Frontend/feature_tracker/models/superpoint_model.pt";
    superPointParam_pin.imageWidth = 352;
    superPointParam_pin.imageHeight = 94;
    superPointParam_pin.distThresh = 2;
    superPointParam_pin.borderRemove = 4;
    superPointParam_pin.confidenceThresh = 0.015;
    superPointParam_pin.gpuIdx = 0;

    superPoint_pin = _cv::SuperPoint::create(superPointParam_pin);

    if (!useOpticalFlow)
    {    
    _cv::SuperGlue::Param superGlueParam;
    superGlueParam.pathToWeights = "/home/wang/catkin_ws/src/MCVO-main-new/MCVO/src/Frontend/feature_tracker/models/superglue_model.pt";
    superGlueParam.gpuIdx = 0;
    superGlue = _cv::SuperGlue::create(superGlueParam);
    LOG(INFO) << "superglue().initial end";
    }
}

void SupFeatureTracker::SortScore(vector<cv::KeyPoint> _pts, cv::Mat _desc, int num)
{
    n_pts.clear();
    n_desc.release();
    std::vector<std::pair<cv::KeyPoint, int>> cnt_id;
    for (unsigned int i = 0; i < _pts.size(); i++)
    cnt_id.push_back(std::make_pair(_pts[i], i));
    sort(cnt_id.begin(), cnt_id.end(), [](const pair<cv::KeyPoint, int> &a, const pair<cv::KeyPoint, int> &b)
        { return a.first.response > b.first.response; });


    for (int i = 0; i < min(int(cnt_id.size()), num); i++)
    {
            n_pts.push_back(cnt_id[i].first);
            // cv::Mat row = forw_desc(cv::Range(it.first.second, it.first.second + 1), cv::Range(0, forw_desc.cols));
            cv::Mat row = _desc.row(cnt_id[i].second);
            if (n_desc.rows == 0)
            {
                n_desc = row;
            }
            else
            {
                cv::vconcat(n_desc, row, n_desc);
            }
        
    }
}
void SupFeatureTracker::setMask()
{
    if (cam->FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(cam->ROW, cam->COL, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time

    vector<pair<pair<int, int>, pair<cv::KeyPoint, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(make_pair(track_cnt[i], i), make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<pair<int, int>, pair<cv::KeyPoint, int>> &a, const pair<pair<int, int>, pair<cv::KeyPoint, int>> &b)
         { return a.first.first > b.first.first; });
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    cv::Mat tmp_desc;

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first.pt) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first.first);
            cv::circle(mask, it.second.first.pt, cam->MIN_DIST, 0, -1);
            // cv::Mat row = forw_desc(cv::Range(it.first.second, it.first.second + 1), cv::Range(0, forw_desc.cols));
            cv::Mat row = forw_desc.row(it.first.second);
            if (tmp_desc.rows == 0)
            {
                tmp_desc = row;
            }
            else
            {
                cv::vconcat(tmp_desc, row, tmp_desc);
            }
        }
    }
    forw_desc.release();
    forw_desc = tmp_desc;
}

void SupFeatureTracker::addPoints()
{
    // for (auto &p : n_pts)
    // {
    //     forw_pts.push_back(p);
    //     ids.push_back(-1);
    //     track_cnt.push_back(1);
    // }
    int count = 0;
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
        // cv::Mat row = n_desc(cv::Range(count, count + 1), cv::Range(0, n_desc.cols));
        cv::Mat row = n_desc.row(count);
        count++;
        if (forw_desc.rows != 0)
        {
            cv::vconcat(forw_desc, row, forw_desc);
        }
        else
        {
            forw_desc = row;
        }
    }
}

void SupFeatureTracker::keypoints(cv::Mat rimage, int num)
{
    std::vector<torch::jit::IValue> input;
    torch::Device device = torch::kCUDA;
    torch::Tensor x = torch::from_blob(rimage.data, {1, 1, rimage.rows, rimage.cols}, at::kByte).toType(at::kFloat).to(device) / 255;
    input.push_back(x);

    // time to inference 
    auto z = model.forward(input).toTuple();
    auto semi = z->elements()[0].toTensor();
    auto desc = z->elements()[1].toTensor();

    //DESCRIPTOR
    auto dn = norm(desc, 2, 1);
    desc = at::div((desc), unsqueeze(dn, 1));

    //DETECTOR - POST PROCESS
    semi = softmax(semi, 1);            // 65개 채널에서 [H/8 * W/8] 개 원소으 총합 1이 되도록 regression.
    semi = semi.slice(1, 0, 64);        // remove rest_bin
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]

    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8}); 
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8}); // [B, H, W]
    semi = semi.squeeze(0);
    at::Tensor kpts = (semi > 0.2);
    if (device == c10::kCUDA)
        kpts = kpts.to(kCPU);
    int rowlen = kpts.size(0);
    int collen = kpts.size(1);

    auto pT1 = kpts.data_ptr<bool>();
    auto pT2 = pT1 + collen;
    // auto pT3 = pT2 + collen;

    for(int i = 0; i < rowlen; i++)
    {
        for(int j = 0 ; j < collen; j++)
        {
            if(*pT1 && (i < rowlen-1) && (j < collen-1))
            {
                *(pT1 + 1) = 0;             // *(pT1 + 2) = 0;
                *pT2 = 0; *(pT2 + 1) = 0;   // *(pT2 + 2) = 0; 
                //*pT3 = 0; *(pT3 + 1) = 0; *(pT3 + 2) = 0; 
            }
            pT1++;
            pT2++;
            // pT3++;
        }
    }
    if (device == c10::kCUDA)
        kpts = kpts.to(kCUDA);
    at::Tensor kpts_scores = semi.masked_select({kpts});

    // int i_num = 1;
    // while (kpts_scores.sizes()[0] < num && i_num < 3)
    // {
    //     kpts = (semi > 0.3 - 0.1*i_num);
    //     kpts_scores = semi.masked_select({kpts});
    //     i_num++;
    // }
    std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(kpts_scores.unsqueeze(1), 0, 1);
    torch::Tensor idx = std::get<1>(sort_ret).squeeze(1);
    kpts = at::nonzero(kpts); // [N, 2] (y, x)

  
    at::Tensor fkpts = kpts.to(kFloat);
    at::Tensor grid = torch::zeros({1, 1, kpts.size(0), 2}).to(device); 
    grid[0][0].slice(1, 0, 1) = (2.0 * (fkpts.slice(1, 1, 2) / semi.size(1))) - 1; // x
    grid[0][0].slice(1, 1, 2) = (2.0 * (fkpts.slice(1, 0, 1) / semi.size(0))) - 1; // y

    torch::Tensor Desc = at::grid_sampler(desc, grid, 0, 0, false);    // [1, 256, 1, n_keypoints]
    Desc = Desc.squeeze(0).squeeze(1);                  // [256, n_keypoints]

    at::Tensor dnn = at::norm(Desc, 2, 1);          // [CUDAFloatType [256]]
    Desc = at::div((Desc + EPSILON), unsqueeze(dnn, 1));
    //mDesc = mDesc.div(unsqueeze(dn, 1));          // [256, n_keypoints] <- unsqueeezed dn[CUDAFloatType [256, 1]]
    Desc = Desc.transpose(0, 1).contiguous();   
    if (device == c10::kCUDA)
        Desc = Desc.to(kCPU);

    cv::Mat _descriptors;
    int n_keypoints = Desc.size(0); 
    // _descriptors.create(n_keypoints, 256, CV_32FC1);
    // memcpy((void*)_descriptors.data, Desc.data_ptr(), sizeof(float) * Desc.numel());

    int size = min(num, n_keypoints);
    torch::Tensor Desc_copy = torch::zeros({size, 256});

    for(int k = 0; k < size; k++)
    {
        int ids = idx[k].item<int>();
        int x = kpts[ids][1].item<int>();
        int y = kpts[ids][0].item<int>();
        float conf = kpts_scores[ids].item<float>();
        // std::cout<< conf<< std::endl;
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt.x = x;
        newKeyPoint.pt.y = y;
        newKeyPoint.response = conf;
        n_pts.push_back(newKeyPoint);
        Desc_copy[k] = Desc[k];
    }
    _descriptors.create(num, 256, CV_32FC1);
    memcpy((void*)_descriptors.data, Desc_copy.data_ptr(), sizeof(float) * Desc_copy.numel());
    n_desc = _descriptors;

}
//接收深度值

void SupFeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{

    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    cout << cam->name << endl;

    // too dark or too bright: histogram
    if (cam->EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        // curr_img<--->forw_img
        prev_img = cur_img = forw_img = img;
        // prev_depth = cur_depth = forw_depth = _depth;
    }
    else
    {
        forw_img = img;
        // forw_depth = _depth;
    }

    forw_pts.clear();
    forw_desc.release();
    vector<cv::KeyPoint> forw_tmp_pts;
    cv::Mat forw_tmp_desc;
cout << cur_pts.size() << endl;
    if (cur_pts.size() > 0)
    {

        TicToc t_o;
        vector<uchar> status;

        if (!useOpticalFlow)
        {

            vector<cv::KeyPoint> tmp_pts, n_tmp_pts;
            cv::Mat tmp_desc, n_tmp_desc, forw_desc_;
            // LOG(INFO) << "Extract points using Superpoint extractor in ORB_SLAM2";
            superPoint->detectAndCompute(forw_img, fisheye_mask, n_tmp_pts, n_tmp_desc);

            SortScore(n_tmp_pts, n_tmp_desc, 1000);

            LOG(INFO) << "new points size" << n_pts.size();

            
            // keypoints(forw_img, 1000);
            tmp_pts = n_pts;
            tmp_desc = n_desc.clone();
            forw_tmp_pts = n_pts;
            forw_tmp_desc = n_desc.clone();

            LOG(INFO) << "START Matching";


            std::vector<cv::DMatch> matches;
            superGlue->match(cur_desc, cur_pts, cur_img.size(), tmp_desc, tmp_pts,
                forw_img.size(), matches);
            float ratio_thresh = 0.75f;
            idx_vector.clear();
            forw_pts.clear();
            forw_desc.release();
            int count = 0;

            for (size_t i = 0; i < cur_pts.size(); i++)
            {
                int idx = 0;
                if (matches[count].queryIdx == i)
                {
                    status.push_back(1);
                    idx = matches[count].trainIdx;
                    idx_vector.push_back(idx);
                    count++;
                }
                else
                {
                    status.push_back(0);
                }
               
                forw_pts.push_back(tmp_pts[idx]); 

                cv::Mat row = tmp_desc.row(idx);
                if (forw_desc_.rows != 0)
                {
                    cv::vconcat(forw_desc, row, forw_desc);
                }
                else
                {
                    forw_desc = row;
                }
            }
            cout << "matches:" << count << endl;
            //                     // my work wang
            // vector<float> err;
            // vector<cv::Point2f> cur_pts_;
            // int count_cur = 0;
            // for (int i = 0; i < status.size(); i++)
            // {
            //     if(status[i])
            //         cur_pts_.push_back(cv::Point2f(cur_pts[i].pt.x, cur_pts[i].pt.y));
            // }
            // vector<cv::Point2f> forw_pts_;
            // vector<uchar> status_;
            // cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts_, forw_pts_, status_, err, cv::Size(21, 21), 3);
            // int count_forw = 0;
            // int num_cur = 0;
            // cv::Mat row;
            // for (int i = 0; i < status.size(); i++)
            // {
            //     if(status[i])
            //     {
            //         int dist = abs(forw_pts[i].pt.x - forw_pts_[count_forw].x) + abs(forw_pts[i].pt.y - forw_pts_[count_forw].y);
            //         // cout << "distence" << dist << endl;
            //         if(dist > 3||!status_[count_forw])
            //         {
            //             status[i] = 0;
            //             row = forw_desc_.row(i);
            //         }
            //         else
            //         {
            //             num_cur++;
            //             forw_pts[i].pt.x = forw_pts_[count_forw].x;
            //             forw_pts[i].pt.y = forw_pts_[count_forw].y;
            //             // forw_pts[i].pt.x = (forw_pts_[count_forw].x + forw_pts[i].pt.x)/2;
            //             // forw_pts[i].pt.y = (forw_pts_[count_forw].y + forw_pts[i].pt.y)/2;
            //             row = cur_desc.row(i);
            //         }
            //         count_forw++;
            //     }
            //     else
            //     {
            //         row = forw_desc_.row(i);
            //     }
            //     if (forw_desc.rows != 0)
            //     {
            //         cv::vconcat(forw_desc, row, forw_desc);
            //     }
            //     else
            //     {
            //         forw_desc = row;
            //     }
            //     row.release();
            // }
            // cout << "1111111111122222222222223333333distence <= 3     " << num_cur << endl;
        }
        else
        {
            vector<float> err;
            vector<cv::Point2f> cur_pts_;
            for (auto &i : cur_pts)
            {
                cur_pts_.push_back(cv::Point2f(i.pt.x, i.pt.y));
            }

            vector<cv::Point2f> forw_pts_;
            cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts_, forw_pts_, status, err, cv::Size(21, 21), 3);
            forw_pts.clear();

            for (auto &i : forw_pts_)
            {
                cv::KeyPoint p;
                p.pt.x = i.x;
                p.pt.y = i.y;
                //p.response = 这里得到分数
                //将分数和追踪次数结合形成匹配标准
                //没有评价匹配效果的指标
                forw_pts.push_back(p);
            }
            forw_desc = cur_desc.clone();
        }

        for (int i = 0; i < int(forw_pts.size()); i++)
            if ((status[i] && !inBorder(forw_pts[i].pt)))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        TrackerBase::reduceVector(ids, status);
        TrackerBase::reduceVector(cur_un_pts, status);
        TrackerBase::reduceVector(track_cnt, status);

        reduceMat(forw_desc, status);
        reduceMat(prev_desc, status);
        reduceMat(cur_desc, status);


        // cout << "forw_pts[i].pt                 21121212121221221111111111111111              cur_pts[i].pt " <<endl;
        // for (int i = 0; i < int(forw_pts.size()); i++)
        // {
        // // cout << "forw_pts[i].pt "<< forw_pts[i].pt <<endl;

        // // cout << "cur_pts[i].pt " <<cur_pts[i].pt <<endl;     

        // cout << "ftrack_cnt[i].pt "<< track_cnt[i] <<endl;
        // }
        
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (cam->PUB_THIS_FRAME)
    {
        //对prev_pts和forw_pts做ransac剔除outlier.

        rejectWithF();
        cout << "after ransac forw_pts "<< forw_pts.size() <<endl;

        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        // int array[20] = {0};
        // for(int i = 0; i < int(track_cnt.size()); i++)
        // {
        //     if (track_cnt[i]<20)
        //     array[track_cnt[i]]++;
        // }
        // for(int i = 0; i < 20; i++)
        // {
        //     cout << "track times" << i  << " : "  << array[i] << endl;
        // }

        ROS_DEBUG("set mask costs %fms", t_m.toc());
        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = cam->MAX_CNT - static_cast<int>(forw_pts.size());

        if (n_max_cnt > 0)
        {
            if (mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            
            n_desc.release();
            n_pts.clear();
         
            if (forw_tmp_pts.size() > 0)
            {
            int count_idx = 0;

            sort(idx_vector.begin(), idx_vector.end());

            for(int i = 0; (int(n_pts.size()) < n_max_cnt)&&(i < forw_tmp_pts.size()); i++)
            {
                if(idx_vector[count_idx] != i)
                {

                    n_pts.push_back(forw_tmp_pts[i]);
                    cv::Mat row = forw_tmp_desc.row(i);
                    if (n_desc.rows != 0)
                    {
                        cv::vconcat(n_desc, row, n_desc);
                    }
                    else
                    {
                        n_desc = row;
                    }

                }
                else
                {
                    count_idx++;
                }

            }
            }
            else
            { 
                vector<cv::KeyPoint> _pts;
                cv::Mat _desc;
                cv::Mat dstimage;
                if(cam->name == "camera_03"||cam->name == "camera_02")
                {
                    superPoint->detectAndCompute(forw_img, fisheye_mask, _pts, _desc);
                }
                else
                {
                    superPoint_pin->detectAndCompute(forw_img, fisheye_mask, _pts, _desc);
                }
                resize(forw_img, dstimage, cv::Size(forw_img.cols/4, forw_img.rows/4), 0, 0, cv::INTER_LINEAR);
                SortScore(_pts, _desc, n_max_cnt);
            }
        }
        else
        {
            n_desc.release();
            n_pts.clear();
        }
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    // prev_depth = cur_depth;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    // cur_depth = forw_depth;
    cur_pts = forw_pts;
    prev_desc = cur_desc;
    cur_desc = forw_desc;
    undistortedPoints();
    prev_time = cur_time;
    
    // LOG(INFO) << "xxxxxxxxxxxxxxxxxxxxxxxxxxTract points using Superpoint extractor in ORB_SLAM2";
}

void SupFeatureTracker::reduceMat(cv::Mat &mat, vector<uchar> &status)
{
    int j = 0;
    if (mat.rows == 0)
        return;
    // LOG(INFO) << mat.rows;
    // LOG(INFO) << status.size();
    // assert(mat.rows == status.size());
    for (int i = 0; i < int(mat.rows); i++)
    {
        if (status[i])
        {
            mat.row(j) -= mat.row(j);
            mat.row(j) += mat.row(i);
            j++;
        }
    }
    mat = mat(cv::Range(0, j), cv::Range::all());
}


void SupFeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].pt.x, cur_pts[i].pt.y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + cam->COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + cam->ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());


            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].pt.x, forw_pts[i].pt.y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + cam->COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + cam->ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }
        // //my code begin
        // vector<cv::Point2f> _cur_pts, _forw_pts;

        // for (auto &i : cur_pts)
        // {
        //     _cur_pts.push_back(cv::Point2f(i.pt.x,i.pt.y));
        // }
        // for (auto &i : forw_pts)
        // {
        //     _forw_pts.push_back(cv::Point2f(i.pt.x,i.pt.y));
        // }
        // //my code end

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, cam->F_THRESHOLD, 0.99, status);//倒数第二个参数原来为0.99， 前两个参数为un_cur_pts, un_forw_pts

        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        TrackerBase::reduceVector(cur_un_pts, status);
        TrackerBase::reduceVector(ids, status);
        TrackerBase::reduceVector(track_cnt, status);//reduceVector就是去掉向量中这一部分值删去，不是变为0
        reduceMat(forw_desc, status);
        reduceMat(prev_desc, status);
        reduceMat(cur_desc, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool SupFeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void SupFeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void SupFeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(cam->ROW + 600, cam->COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < cam->COL; i++)
        for (int j = 0; j < cam->ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + cam->COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + cam->ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        // cout << trackerData[0].K << endl;
        // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < cam->ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < cam->COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            // ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void SupFeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    // cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].pt.x, cur_pts[i].pt.y);
        Eigen::Vector3d b;
        // https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/0d280936e441ebb782bf8855d86e13999a22da63/camera_model/src/camera_models/PinholeCamera.cc
        // brief Lifts a point from the image plane to its projective ray
        m_camera->liftProjective(a, b);
        // 特征点在相机坐标系的归一化坐标
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

void SupFeatureTracker::getPt(int idx, int &id, geometry_msgs::Point32 &p, cv::Point2f &p_uv, cv::Point2f &v)
{   
    id = ids[idx];
    p.x = cur_un_pts[idx].x;
    p.y = cur_un_pts[idx].y;
    p.z = 1;
    p_uv.x = cur_pts[idx].pt.x;
    p_uv.y = cur_pts[idx].pt.y;
    v = pts_velocity[idx];//应该只改这一部分
}

void SupFeatureTracker::getCurPt(int idx, cv::Point2f &cur_pt)
{
    cur_pt.x = cur_pts[idx].pt.x;
    cur_pt.y = cur_pts[idx].pt.y;
}
