#include "keyframe.h"


template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, vector<cv::Mat> &_image_list, vector<int> _point_num,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id, int _sequence)
{
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;



	for(int i = 0; i < _image_list.size(); i++)
	{
		image_list.push_back(_image_list[i].clone());
		point_num.push_back(_point_num[i]);
	}
	cv::resize(image_list[0], thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;

	computeWindowBRIEFPoint();//变为计算四个的

	computeBRIEFPoint();
	
	// if(!DEBUG_IMAGE)
	// 	image_list.clear();
	
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
					cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
					vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors)
{
	time_stamp = _time_stamp;
	index = _index;
	//vio_T_w_i = _vio_T_w_i;
	//vio_R_w_i = _vio_R_w_i;
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = _T_w_i;
	R_w_i = _R_w_i;
	if (DEBUG_IMAGE)
	{
		image_list[0] = _image.clone();
		cv::resize(image_list[0], thumbnail, cv::Size(80, 60));
	}
	if (_loop_index != -1)
		has_loop = true;
	else
		has_loop = false;
	loop_index = _loop_index;
	loop_info = _loop_info;
	has_fast_point = false;
	sequence = 0;
	keypoints = _keypoints;
	keypoints_norm = _keypoints_norm;
	brief_descriptors = _brief_descriptors;
}


void KeyFrame::computeWindowBRIEFPoint()
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
		cv::KeyPoint key;
		key.pt = point_2d_uv[i];
		window_keypoints.push_back(key);
	}
	int point_index = 0;
	for(int i = 0; i < (int)point_num.size(); i++)
	{
		vector<BRIEF::bitset> one_brief;
		vector<cv::KeyPoint> one_keypoints;

		for(int j = 0; j < point_num[i]; j++)
		{
			one_keypoints.push_back(window_keypoints[point_index]);
			point_index++;
		}
		extractor(image_list[i], one_keypoints, one_brief);
		compute_point_num.push_back((int)one_keypoints.size());
		for(int j =0; j < (int)one_brief.size(); j++)
			window_brief_descriptors.push_back(one_brief[j]);
	}
}

void KeyFrame::computeBRIEFPoint()
{
	cv::Mat im;

	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
	for(int j = 0; j < (int)point_num.size(); j++)
	{
		const int fast_th = 20; // corner detector response threshold
		vector<BRIEF::bitset> one_brief;
		vector<cv::KeyPoint> one_keypoints;
		vector<cv::KeyPoint> one_keypoints_norm;

		if(0)
			cv::FAST(image_list[j], one_keypoints, fast_th, true);
		else
		{
			vector<cv::Point2f> tmp_pts;
			cv::goodFeaturesToTrack(image_list[j], tmp_pts, 500, 0.01, 10);
			for(int i = 0; i < (int)tmp_pts.size(); i++)
			{
				cv::KeyPoint key;
				key.pt = tmp_pts[i];
				one_keypoints.push_back(key);
			}
		}
		extractor(image_list[j], one_keypoints, one_brief);
		for (int i = 0; i < (int)one_keypoints.size(); i++)
		{
			Eigen::Vector3d tmp_p;
			m_camera_vector[j]->liftProjective(Eigen::Vector2d(one_keypoints[i].pt.x, one_keypoints[i].pt.y), tmp_p); 
			cv::KeyPoint tmp_norm;
			tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
			one_keypoints_norm.push_back(tmp_norm);
		}

		for(int i =0; i < (int)one_brief.size(); i++)
		{
			keypoints.push_back(one_keypoints[i]);
			brief_descriptors.push_back(one_brief[i]);
			keypoints_norm.push_back(one_keypoints_norm[i]);
		}

	}

}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
							const cv::KeyPoint window_keypoint,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
							const std::vector<int> &old_point_camera_vector,
							int &old_point_camera,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < descriptors_old.size(); i++)
    {
		int dis = HammingDis(window_descriptor, descriptors_old[i]);
		if(dis < bestDist)
		{
			bestDist = dis;
			bestIndex = i;
		}

    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
		best_match = keypoints_old[bestIndex].pt;
		best_match_norm = keypoints_old_norm[bestIndex].pt;
		int points_num = -1;
		for(int j = 0; j < old_point_camera_vector.size(); j++)
		{
			old_point_camera = j;	
			if(j == old_point_camera_vector.size() - 1)
				break;
			else if(bestIndex > points_num&& bestIndex <= points_num + old_point_camera_vector[j])
				break;
			points_num += old_point_camera_vector[j];
		}
	//   cout << " " << old_point_camera;
	//   cout << " " << bestIndex;
      return true;
    }
    else
      return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
								std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
								std::vector<int> &point_num_vector,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm,
								const std::vector<int> &old_point_camera_vector)
{


    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {

        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
		int old_point_camera = -1;

        if (searchInAera(window_brief_descriptors[i], window_keypoints[i], descriptors_old, keypoints_old, keypoints_old_norm, old_point_camera_vector, old_point_camera, pt, pt_norm))
          status.push_back(1);
        else
          status.push_back(0);

        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
		point_num_vector.push_back(old_point_camera);
    }

}

bool KeyFrame::CompareSimilarity(vector<cv::Mat> &image1, vector<cv::Mat> &image2)
{

	bool sim_flag = true;


	for(int i = 0; i < image1.size(); i++)
	{		
		cv::Mat hist1, hist2;
		int histSize = 256;
		float range[] = {0, 256};
		const float* histRange = {range};
		cv::calcHist(&image1[i], 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange);
		cv::calcHist(&image2[i], 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange);

		// 归一化直方图
		cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

		// 计算直方图的相关性
		double correlation = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
		if(correlation < 0.98)
		{
			sim_flag = false;
			break;
		}

	}

    // 计算 SSIM


    return sim_flag;
}
void KeyFrame::OpticalMatch(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
						  std::vector<int> &old_point_camera_vector,
						  const std::vector<int> &point_num_vector,
						  const std::vector<cv::Mat> &old_image_list)
{
	int num = 0;
	for(int i = 0; i < (int)image_list.size(); i++)
	{
		vector<float> err;
		vector<uchar> pre_status;
		vector<cv::Point2f> cur_pts_;
		vector<cv::Point2f> old_pts_;
		for (int j = 0; j < point_num[i]; j++)
		{
			cur_pts_.push_back(cv::Point2f(window_keypoints[j + num].pt.x, window_keypoints[j + num].pt.y));
		}
		num += (int)cur_pts_.size();
		cv::calcOpticalFlowPyrLK(image_list[i], old_image_list[i], cur_pts_, old_pts_, pre_status, err, cv::Size(21, 21), 3);
		for(int j = 0; j < old_pts_.size(); j++)
		{
			matched_2d_old.push_back(old_pts_[j]);
			Eigen::Vector3d tmp_p;
			m_camera_vector[i]->liftProjective(Eigen::Vector2d(old_pts_[j].x, old_pts_[j].y), tmp_p); 
			cv::Point2f tmp_norm;
			tmp_norm = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
			matched_2d_old_norm.push_back(tmp_norm);
			status.push_back(pre_status[j]);
			old_point_camera_vector.push_back(i);
		}
		
	}
}

void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
	int n = (int)matched_2d_cur_norm.size();
	for (int i = 0; i < n; i++)
		status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
            double FOCAL_LENGTH = 460.0;
            double tmp_x, tmp_y;
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old, int camera_index, KeyFrame* old_kf)
{
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = old_kf->origin_vio_R * qic_vector[camera_index];
    Vector3d T_w_c = old_kf->origin_vio_T + old_kf->origin_vio_R * tic_vector[camera_index];


    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;
    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
		{
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 /460.0, 0.99, inliers);

		}
    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }


    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic_vector[camera_index].transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic_vector[camera_index];

}

void KeyFrame::PnPRANSAC_op(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old, int camera_index, KeyFrame* old_kf, Eigen::Vector3d &PnP_T_op, Eigen::Matrix3d &PnP_R_op)
{
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
	Matrix3d R_w_c;
    Vector3d T_w_c;
	if(camera_index == 0)
	{
		R_w_c = old_kf->origin_vio_R * qic_vector[camera_index];
    	T_w_c = old_kf->origin_vio_T + old_kf->origin_vio_R * tic_vector[camera_index];

	}
	else
	{
		R_w_c = PnP_R_op * qic_vector[camera_index];
    	T_w_c = PnP_T_op + PnP_R_op * tic_vector[camera_index];
	}


    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;
    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
		{
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 /460.0, 0.99, inliers);

		}
    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(1);


    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic_vector[camera_index].transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic_vector[camera_index];

}


bool KeyFrame::findConnection(KeyFrame* old_kf)
{
	TicToc tmp_t;

	//printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur, matched_2d_old;
	vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;
	vector<uchar> status;
	bool matched_flag = false;
	bool matched_flag_brief = false;
	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;


	TicToc t_match;
	#if 0
		if (DEBUG_IMAGE)    
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	//printf("search by des\n");
	for(int i = 0; i < point_num.size(); i++)
	{
		for(int j = 0; j < point_num[i]; j++)
			point_num_vector.push_back(i);
	}
	

	 
	std::vector<int> old_point_camera_vector;
	if(CompareSimilarity(image_list, old_kf->image_list))
	{
		OpticalMatch(matched_2d_old, matched_2d_old_norm, status, old_point_camera_vector, point_num, old_kf->image_list);
	
		reduceVector(matched_2d_cur, status);
		reduceVector(matched_2d_old, status);
		reduceVector(matched_2d_cur_norm, status);
		reduceVector(matched_2d_old_norm, status);
		reduceVector(matched_3d, status);
		reduceVector(matched_id, status);
		reduceVector(point_num_vector, status);
		if((int)matched_2d_cur.size() >0.5*point_num.size()* MIN_LOOP_NUM)
		 	matched_flag = true;
	}
	else
	{
		if((int)matched_2d_cur.size() > point_num.size()* MIN_LOOP_NUM)
		{
			searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_point_camera_vector, old_kf->window_brief_descriptors, old_kf->window_keypoints, old_kf->keypoints_norm, old_kf->compute_point_num);
			reduceVector(old_point_camera_vector, status);
			reduceVector(matched_2d_cur, status);
			reduceVector(matched_2d_old, status);
			reduceVector(matched_2d_cur_norm, status);
			reduceVector(matched_2d_old_norm, status);
			reduceVector(matched_3d, status);
			reduceVector(matched_id, status);
			reduceVector(point_num_vector, status);
			if((int)matched_2d_cur.size() > point_num.size()* MIN_LOOP_NUM)
			{
				matched_flag = true;
				matched_flag_brief = true;
			}
		}
	}
	

	#if 0 
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);	        
	        */
	        
	    }
	#endif
	status.clear();
	/*
	FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);
	*/
	#if 0
		if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
	#endif
	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;
	double relative_pitch;
	double relative_roll;


	if(matched_flag)
	{
		matched_flag = false;
		status.clear();
		vector<vector<cv::Point2f>> cameras_2d_old_norm;
		vector<vector<cv::Point2f>> cameras_2d_now_norm;
		vector<vector<cv::Point2f>> cameras_2d_old;
		vector<vector<cv::Point2f>> cameras_2d_now;
		vector<vector<cv::Point3f>> cameras_3d;
		
		for(int i = 0; i < (int)point_num.size(); i++)
		{
			vector<cv::Point2f> points_2d_old_norm;
			vector<cv::Point2f> points_2d_now_norm;
			vector<cv::Point3f> points_3d;
			cameras_2d_old_norm.push_back(points_2d_old_norm);
			cameras_2d_now_norm.push_back(points_2d_now_norm);
			cameras_2d_old.push_back(points_2d_old_norm);
			cameras_2d_now.push_back(points_2d_now_norm);
			cameras_3d.push_back(points_3d);
		}

		for(int i = 0; i < (int)point_num_vector.size(); i++)
		{
			// cout << "  " << point_num_vector[i] ;
			if(matched_flag_brief == false)
			{
				cameras_2d_old_norm[point_num_vector[i]].push_back(matched_2d_old_norm[i]);
				cameras_3d[point_num_vector[i]].push_back(matched_3d[i]);
				cameras_2d_now_norm[point_num_vector[i]].push_back(matched_2d_cur_norm[i]);
				cameras_2d_old[point_num_vector[i]].push_back(matched_2d_old[i]);
				cameras_2d_now[point_num_vector[i]].push_back(matched_2d_cur[i]);
			}
			else
			{
				cameras_2d_old_norm[old_point_camera_vector[i]].push_back(matched_2d_old_norm[i]);
				cameras_3d[old_point_camera_vector[i]].push_back(matched_3d[i]);
				cameras_2d_old[old_point_camera_vector[i]].push_back(matched_2d_old[i]);
				matched_flag_brief = false;
			}
		}


		#if (1&&matched_flag_brief == false)
			for(int j = 0; j < image_list.size(); j++)
			{
				
				int gap = 10;
				cv::Mat gap_image(image_list[j].rows, gap, CV_8UC1, cv::Scalar(255, 255, 255));
				cv::Mat gray_img, loop_match_img;
				cv::Mat old_img = old_kf->image_list[j];
				cv::hconcat(image_list[j], gap_image, gap_image);
				cv::hconcat(gap_image, old_img, gray_img);
				cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
				for(int i = 0; i< (int)cameras_2d_now[j].size(); i++)
				{
					cv::Point2f cur_pt = cameras_2d_now[j][i];
					cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
				}
				for(int i = 0; i< (int)cameras_2d_old[j].size(); i++)
				{
					cv::Point2f old_pt = cameras_2d_old[j][i];
					old_pt.x += (image_list[j].cols + gap);
					cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
				}
				for (int i = 0; i< (int)cameras_2d_now[j].size(); i++)
				{
					cv::Point2f old_pt = cameras_2d_old[j][i];
					old_pt.x +=  (image_list[j].cols + gap) ;
					cv::line(loop_match_img, cameras_2d_now[j][i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
				}

				ostringstream path;
				path <<  "/home/wang/Documents/MCVO/MCVO_data_test/image/"
						<< index << "-"
						<< old_kf->index << "- "<< j << " - " << "2fundamental_match.jpg";
				cv::imwrite( path.str().c_str(), loop_match_img);
			}
	    
	#endif

		vector<vector<uchar>> status_4_cameras;
		
		Eigen::Quaterniond quaternion;
		Eigen::Vector3d PnP_T_op;
		Eigen::Matrix3d PnP_R_op;
		bool first_pnp = true;
		int success_camera_num = 0;
		for(int i = 0; i < (int)point_num.size(); i++)
		{
			if((int)cameras_2d_old_norm[i].size() > 8)
			{
				Eigen::Vector3d PnP_T_;
				Eigen::Matrix3d PnP_R_;
				vector<uchar> status_camera;

				PnPRANSAC(cameras_2d_old_norm[i], cameras_3d[i], status_camera, PnP_T_, PnP_R_, i, old_kf);

				int ransac_num = 0;
				for(int j = 0; j < (int)status_camera.size(); j++)
				{
					if((int)status_camera[j] == 1)
						ransac_num++;
				}
				status_4_cameras.push_back(status_camera);
				Eigen::Quaterniond quaternion_n(PnP_R_);

				if(ransac_num > 8)
				{
					success_camera_num++;
					if(first_pnp)
					{
						PnP_T_old = PnP_T_;
						quaternion = quaternion_n;
						first_pnp = false;
					}
					else
					{
						PnP_T_old = PnP_T_old  + PnP_T_;
						quaternion.w() = quaternion.w() + quaternion_n.w();
						quaternion.x() = quaternion.x() + quaternion_n.x();
						quaternion.y() = quaternion.y() + quaternion_n.y();
						quaternion.z() = quaternion.z() + quaternion_n.z();
					}
				}

			}
			else
			{
				vector<uchar> status_camera;
				for(int j = 0; j < (int)cameras_2d_old_norm[i].size(); j++)
					status_camera.push_back(0);
				status_4_cameras.push_back(status_camera);
								
			}
		}

		PnP_T_old = PnP_T_old/success_camera_num;
		quaternion.w() /= success_camera_num;
		quaternion.x() /= success_camera_num;
		quaternion.y() /= success_camera_num;
		quaternion.z() /= success_camera_num;

    	PnP_R_old = quaternion.normalized().toRotationMatrix();

		Quaterniond Q1(PnP_R_old);
		Quaterniond Q2(origin_vio_R);
		Quaterniond Q3(old_kf->origin_vio_R);


		for(int i = 0; i < (int)point_num.size(); i++)
		{
			for(int j = 0; j < (int)status_4_cameras[i].size(); j++)
			status.push_back(status_4_cameras[i][j]);
		}

	    reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);
		reduceVector(point_num_vector, status);

		if((int)matched_2d_cur.size() >0.5*point_num.size()* MIN_LOOP_NUM && success_camera_num > point_num.size()/2)
		 	matched_flag = true;


	    #if 1
	    	if (DEBUG_IMAGE)
	        {
	        	int gap = 10;
	        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
	            cv::Mat gray_img, loop_match_img;
	            cv::Mat old_img = old_kf->image_list[0];
	            cv::hconcat(image_list[0], gap_image, gap_image);
	            cv::hconcat(gap_image, old_img, gray_img);
	            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f cur_pt = matched_2d_cur[i];
	                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for(int i = 0; i< (int)matched_2d_old.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap);
	                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	            }
	            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap) ;
	                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
	            }
	            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
	            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

	            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
	            cv::vconcat(notation, loop_match_img, loop_match_img);

	            if ((int)matched_2d_cur.size() > 0.5*point_num.size()* MIN_LOOP_NUM)
	            {

	            	cv::Mat thumbimage;
	            	cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
	    	    	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
	                msg->header.stamp = ros::Time(time_stamp);
	    	    	pub_match_img.publish(msg);
	            }
	        }
	    #endif
	}

	if (matched_flag)
	{
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    relative_q = PnP_R_old.transpose() * origin_vio_R;
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());
		relative_pitch = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).y() - Utility::R2ypr(PnP_R_old).y());
        relative_roll = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).z() - Utility::R2ypr(PnP_R_old).z());
		
		if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0 )
	    {

	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
			R_cur_in_old = old_kf->origin_vio_R*PnP_R_old.transpose() * origin_vio_R;
			T_cur_in_old = old_kf->origin_vio_R*relative_t + old_kf->origin_vio_T;
	    	if(FAST_RELOCALIZATION)
	    	{
			    sensor_msgs::PointCloud msg_match_points;
			    msg_match_points.header.stamp = ros::Time(time_stamp);

				Eigen::Vector3d T = old_kf->T_w_i; 
			    Eigen::Matrix3d R = old_kf->R_w_i;
			    Quaterniond Q(R);
				Eigen::Vector3d T_ = PnP_T_old; 
			    Eigen::Matrix3d R_ = PnP_R_old;
			    Quaterniond Q_(R_);
			    sensor_msgs::ChannelFloat32 t_q_index;
			    t_q_index.values.push_back(T.x());
			    t_q_index.values.push_back(T.y());
			    t_q_index.values.push_back(T.z());
			    t_q_index.values.push_back(Q.w());
			    t_q_index.values.push_back(Q.x());
			    t_q_index.values.push_back(Q.y());
			    t_q_index.values.push_back(Q.z());
			    t_q_index.values.push_back(index);
				t_q_index.values.push_back(T_.x());
			    t_q_index.values.push_back(T_.y());
			    t_q_index.values.push_back(T_.z());
			    t_q_index.values.push_back(Q_.w());
			    t_q_index.values.push_back(Q_.x());
			    t_q_index.values.push_back(Q_.y());
			    t_q_index.values.push_back(Q_.z());
				msg_match_points.channels.push_back(t_q_index);

			    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
			    {
					sensor_msgs::ChannelFloat32 point_camera_;
		            geometry_msgs::Point32 p;
		            p.x = matched_2d_old_norm[i].x;
		            p.y = matched_2d_old_norm[i].y;
		            p.z = matched_id[i];
		            msg_match_points.points.push_back(p);
					point_camera_.values.push_back(point_num_vector[i]);

					point_camera_.values.push_back(point_num_vector[i]);
					msg_match_points.channels.push_back(point_camera_);
			    }
			   
			    pub_match_points.publish(msg_match_points);
	    	}
	        return true;
	    }
	}
	return false;
}


int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::getcurPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_cur_in_old;
    _R_w_i = R_cur_in_old;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}


