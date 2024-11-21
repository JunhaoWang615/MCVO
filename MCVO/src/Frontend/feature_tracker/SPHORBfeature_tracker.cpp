#include "SPHORBfeature_tracker.h"

#include <fstream>
using namespace MCVO;
bool SPHORBFeatureTracker::inBorder(const cv::Point2f &pt)
{
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < cam->COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < cam->ROW - BORDER_SIZE;
}

void SPHORBFeatureTracker::reduceVector(vector<cv::KeyPoint> &v, vector<uchar> &status)
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

void SPHORBFeatureTracker::reduceMat(cv::Mat &mat, vector<uchar> &status)
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

SPHORBFeatureTracker::SPHORBFeatureTracker()
{
    matcher = BFMatcher(cv::NORM_HAMMING, false);
    max_dist = 30;
    sorb = std::make_shared<cv::SPHORB>(cv::SPHORB());
}

SPHORBFeatureTracker::SPHORBFeatureTracker(std::string rootPath)
{
    matcher = BFMatcher(cv::NORM_HAMMING, false);
    max_dist = 30;
    sorb = std::make_shared<cv::SPHORB>(cv::SPHORB(rootPath));
}

SPHORBFeatureTracker::~SPHORBFeatureTracker()
{
    // sorb->releaseMemory();
}

void SPHORBFeatureTracker::setMask()
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
            cv::Mat row = forw_desc(cv::Range(it.first.second, it.first.second + 1), cv::Range(0, forw_desc.cols));
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

void SPHORBFeatureTracker::addPoints()
{
    // cout << "add pts" << endl;
    // cout << "forw_desc:" << forw_desc.size() << endl;
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
    // cout << "finish adding" << endl;
    // cout << "forw_desc:" << forw_desc.size() << endl;
}

void SPHORBFeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    // convert to gray scale
    // LOG(INFO)<<"image type: " <<img.type();
    // cv::cvtColor(_img, img, CV_BGR2GRAY);
    if (cam->EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    // LOG(INFO) << "OpenCV version : " << CV_VERSION;
    // LOG(INFO) << "Major version : " << CV_MAJOR_VERSION;
    // LOG(INFO) << "Minor version : " << CV_MINOR_VERSION;
    // LOG(INFO) << "Subminor version : " << CV_SUBMINOR_VERSION;
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();
    forw_desc.release();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        if (!useOpticalFlow)
        {
            // vector<float> err;
            // vector<cv::DMatch> matches;
            // std::vector<std::vector<cv::DMatch>> dupMatches;
            vector<Matches> dupMatches;
            // SPHORBdetector.setMaxFeatures(500);
            vector<cv::KeyPoint> tmp_pts;
            cv::Mat tmp_desc;
            LOG(INFO) << "Call sorb detect at matching";
            (*sorb)(forw_img, Mat(), tmp_pts, tmp_desc);
            // ORBdetector->detectAndCompute(forw_img, fisheye_mask, tmp_pts, tmp_desc);
            // ORBdetector->detect(forw_img, n_pts);
            // ORBdetector->compute(forw_img, n_pts, n_desc);

            // if (cur_desc.empty())
            //     cv::error(0, "MatchFinder", "cur desc empty", __FILE__, __LINE__);
            // if (tmp_desc.empty())
            //     cv::error(0, "MatchFinder", "tmp desc empty", __FILE__, __LINE__);
            // cerr << "matching" << endl;
            // cur_desc.convertTo(cur_desc, 5);
            // tmp_desc.convertTo(tmp_desc, 5);
            cerr << cur_desc.rows << endl;
            cerr << tmp_desc.rows << endl;
            matcher.knnMatch(cur_desc, tmp_desc, dupMatches, 2);
            // std::vector<cv::DMatch> good_matches;
            float ratio_thresh = 0.75f;
            forw_pts.clear();
            forw_desc.release();
            int count = 0;
            // cout << "dupMatches:" << dupMatches.size() << endl;
            // cout << "cur_pts:" << cur_pts.size() << endl;
            for (size_t i = 0; i < dupMatches.size(); i++)
            {
                // cout<<dupMatches[i][0].trainIdx<<endl;
                // cout<<"match 0:"<<dupMatches[i][0].distance<<endl;
                // cout<<"match 1:"<<dupMatches[i][1].distance<<endl;
                if (dupMatches[i][0].distance < ratio_thresh * dupMatches[i][1].distance)
                {

                    // good_matches.push_back(dupMatches[i][0]);
                    count++;
                    status.push_back(1);
                }
                else
                {
                    status.push_back(0);
                }
                int idx = dupMatches[i][0].trainIdx;
                forw_pts.push_back(tmp_pts[idx]);
                cv::Mat row = tmp_desc.row(idx);
                if (forw_desc.rows != 0)
                {
                    cv::vconcat(forw_desc, row, forw_desc);
                }
                else
                {
                    forw_desc = row;
                }
            }
            cout << "matches: " << count << endl;
            // drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
            //      Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            // for (int j = 0; j < forw_desc.rows; j++)
            // {
            //     if (matches[j].distance < max_dist)
            //     {
            //         status.push_back(1);
            //     }
            //     else
            //     {
            //         status.push_back(0);
            //     }
            // }
            // cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
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
                forw_pts.push_back(p);
            }
            forw_desc = cur_desc.clone();
        }
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i].pt))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        TrackerBase::reduceVector(ids, status);
        TrackerBase::reduceVector(cur_un_pts, status);
        TrackerBase::reduceVector(track_cnt, status);
        // cout << "reduce prev_desc" << endl;
        reduceMat(prev_desc, status);
        // cout << "reduce cur_desc" << endl;
        reduceMat(cur_desc, status);
        // cout << "reduce forw_desc" << endl;
        reduceMat(forw_desc, status);
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (cam->PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        n_max_cnt = cam->MAX_CNT - static_cast<int>(forw_pts.size());
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
            // cout << "forw_pts size:" << forw_pts.size() << endl;
            // cout << "Find " << n_max_cnt << " new pts" << endl;
            // ORBdetector->setMaxFeatures(n_max_cnt);
            vector<KeyPoint> tmp_pts;
            Mat tmp_desc;
            tmp_pts.clear();
            tmp_desc.release();
            LOG(INFO) << "Call sorb at detection";
            (*sorb)(forw_img, Mat(), tmp_pts, tmp_desc);
            // int i=100000;
            // while (i--);
            cout << "Finish detecting" << endl;
            n_pts = tmp_pts;
            n_desc = tmp_desc;
            // n_desc.convertTo(n_desc, 5);
            // ORBdetector->detect(forw_img, n_pts);
            // ORBdetector->compute(forw_img, n_pts, n_desc);
            cout << "n_pts size:" << n_pts.size() << endl;
            cout << "n_desc size:" << n_desc.size() << endl;
            // cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
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
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    prev_desc = cur_desc;
    cur_desc = forw_desc;
    undistortedPoints();
    prev_time = cur_time;
}

void SPHORBFeatureTracker::rejectWithF()
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
        // string filenamebase = "/home/yao/pts_test";
        // string filename = filenamebase + "/pts" + to_string(filecount)+".csv";
        // filecount++;
        // ofstream ofile;

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, cam->F_THRESHOLD, 0.99, status);
        // RANSAC_cuda_tools::findFundamentalMat_on_cuda(un_cur_pts,un_forw_pts,0.1,0.99,status);
        // ofile.open(filename);
        // // format: cur pts, forw pts, status
        // for (int i=0;i<un_cur_pts.size();i++){
        //     ofile<<un_cur_pts[i].x<<','<<un_cur_pts[i].y<<','<<un_forw_pts[i].x<<','<<un_forw_pts[i].y<<','<<(int)status[i]<<'\n';
        // }
        // ofile.close();
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        TrackerBase::reduceVector(cur_un_pts, status);
        TrackerBase::reduceVector(ids, status);
        TrackerBase::reduceVector(track_cnt, status);
        // cout << "reduce forw_desc" << endl;
        reduceMat(forw_desc, status);
        // cout << "reduce prev_desc" << endl;
        reduceMat(prev_desc, status);
        // cout << "reduce cur_desc" << endl;
        reduceMat(cur_desc, status);
    }
}

bool SPHORBFeatureTracker::updateID(unsigned int i)
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

void SPHORBFeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void SPHORBFeatureTracker::showUndistortion(const string &name)
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

void SPHORBFeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    // cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].pt.x, cur_pts[i].pt.y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
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

void SPHORBFeatureTracker::getPt(int idx, int &id, geometry_msgs::Point32 &p, cv::Point2f &p_uv, cv::Point2f &v)
{
    id = ids[idx];
    p.x = cur_un_pts[idx].x;
    p.y = cur_un_pts[idx].y;
    p.z = 1;
    p_uv.x = cur_pts[idx].pt.x;
    p_uv.y = cur_pts[idx].pt.y;
    v = pts_velocity[idx];
}

void SPHORBFeatureTracker::getCurPt(int idx, cv::Point2f &cur_pt)
{
    cur_pt.x = cur_pts[idx].pt.x;
    cur_pt.y = cur_pts[idx].pt.y;
}