//Author: Zhuo Su, in Beihang University (BUAA)
//date: 04/2017


#include"InitValue.h"
#include"permutohedral.h"
#include"vincent11.c"
#include<map>
#include<algorithm>
#include<functional>

#define MAX_IMG_DIM 300
#define TOLERANCE 0.01
#define FRAME_MAX 20
#define SOBEL_THRESH 0.4


void InitValue::GetBgvalue(cv::Mat& unaryMap, cv::Mat& unaFuse, const cv::Mat& im, bool usePixel)
{
	//string pic = "..\\..\\MSRA10K_Imgs_GT\\Imgs\\938.jpg";
	int spcount = 300;//300
	double compactness = 20.0;//20
	this->m_info.GetInfomation(im, spcount, compactness);

	getIdxs();

#pragma region illustrate the border
	//illustate the border
	//cv::Mat img1 = cv::imread(pic);
	//cv::Mat img2 = img1.clone();
	//
	//for (auto i : m_borderIdx)
	//{
	//	for (auto bg = m_info.sps_[i].begin(); bg < m_info.sps_[i].end(); bg++)
	//	{
	//		for (int j = 0; j < 3; j++)
	//			img1.at<cv::Vec3b>((*bg).y, (*bg).x)[j] = i;
	//	}
	//}

	//for (auto i : m_borderIdx2)
	//{
	//	for (auto bg = m_info.sps_[i].begin(); bg < m_info.sps_[i].end(); bg++)
	//	{
	//		for (int j = 0; j < 3; j++)
	//			img2.at<cv::Vec3b>((*bg).y, (*bg).x)[j] = i;
	//	}
	//}

	//cv::imshow("border1", img1);
	//cv::imshow("border2", img2);
	//cv::waitKey(0);
#pragma endregion


	//getSalFromClusteredBorder(unaryMap);
	getSalFromGmmBorder(unaryMap, unaFuse, usePixel);

	return;
}

void InitValue::getIdxs(bool indx2)
{
	if (m_borderIdx.size() > 0 || m_innerIdx.size() > 0) return;

	for (int i = 0; i < m_info.numlabels_; i++)
	{
		if (m_info.features_[i].isborder_)
		{
			m_borderIdx.push_back(i);
		}
		else
		{
			m_innerIdx.push_back(i);
		}
	}
	
	if (indx2)
	{
		getIdxs2();
	}
}

void InitValue::getIdxs2()
{
	assert(m_borderIdx.size() > 0);
	assert(m_info.nb_&&!m_info.nbCnt_);

	m_info.getNeighbor(m_info.labelsbuf_);

	m_borderIdx2.clear();
	for (auto i : m_borderIdx)
	{
		for (auto j : m_info.features_[i].neighbor_)
		{
			m_borderIdx2.insert(j);
		}
	}
}

void InitValue::getNeighborCnt()
{
	//m_info.getNeighborCnt();

	if (m_info.nbCnt_) return;
	m_info.getNeighbor(m_info.labelsbuf_);

	for (auto i : m_borderIdx)
	{
		for (auto j : m_borderIdx)
		{
			if (j != i)
			{
				m_info.features_[i].neighbor_.insert(j);
				m_info.features_[j].neighbor_.insert(i);
			}
		}
	}

	for (int i = 0; i < m_info.numlabels_; i++)
	{
		m_info.features_[i].neighborCnt_.clear();
		for (auto j : m_info.features_[i].neighbor_)
		{
			m_info.features_[i].neighborCnt_.insert(j);
			for (auto k : m_info.features_[j].neighbor_)
			{
				if (k != i)
				{
					m_info.features_[i].neighborCnt_.insert(k);
				}
			}
			
		}
	}

	//for (int i = 0; i < m_info.numlabels_; i++)
	//{
	//	m_info.features_[i].neighbor33_.clear();
	//	for (auto j : m_info.features_[i].neighborCnt_)
	//	{
	//		for (auto k : m_info.features_[j].neighbor_)
	//		{
	//			m_info.features_[i].neighbor33_.insert(k);
	//		}
	//	}
	//}

	m_info.nbCnt_ = true;
}

void InitValue::clusterBorder(cv::Mat& borderlabels, std::vector<cv::Vec3f>& border)
{
	//for (int i = 0; i < m_info.numlabels; i++)
	//{
	//	if (m_info.features[i][6])
	//	{
	//		border.push_back(cv::Vec3f(m_info.features[i][0],
	//			m_info.features[i][1], m_info.features[i][2]));
	//	}
	//}

	//CV_Assert(!border.empty());
	//cv::Mat _bgdSamples((int)border.size(), 3, CV_32FC1, &border[0][0]);
	//kmeans(_bgdSamples, 3, borderlabels,
	//	cv::TermCriteria(CV_TERMCRIT_ITER, 10, 0.0), 0, cv::KMEANS_PP_CENTERS);
}

int InitValue::removeCluster(double sumclus[3], cv::Mat& borderlabels, std::vector<cv::Vec3f>& border)
{
	//std::array<std::array<int, 256>, 3> hist[3];
	//memset(sumclus, 0, 3*sizeof(double));//sizeof(sumclus)=8,not 24
	////double sumclus[3] = { 0.0, 0.0, 0.0 };
	//for (int i = 0; i < 3; i++)
	//{
	//	for (auto ite = hist[i].begin(); ite < hist[i].end(); ite++)
	//		ite->assign(0);
	//}
	//const cv::Mat& img = m_info.inputImg;
	//for (int i = 0; i < borderIdx.size(); i++)
	//{
	//	const int clus = borderlabels.at<int>(i, 0);
	//	for (auto ite = m_info.sps[borderIdx[i]].begin(); ite < m_info.sps[borderIdx[i]].end(); ite++)
	//	{
	//		for (int j = 0; j < 3; j++)//L a b color
	//		{
	//			hist[clus][j][img.at<cv::Vec3b>((*ite).y, (*ite).x)[j]]++;
	//		}
	//	}
	//}

	////and then, calculate the inter-distance between each cluster pair
	////this code may be accelerated
	//cv::Mat dis3x3 = cv::Mat_<double>(3, 3);
	//for (int i = 0; i < 3; i++)
	//{
	//	for (int j = i; j < 3; j++)
	//	{
	//		if (i == j) dis3x3.at<double>(i, j) = 0;
	//		else
	//		{
	//			double dis = 0.0;
	//			for (int k = 0; k < 256; k++)
	//			{
	//				dis += pow(hist[i][0][k] - hist[j][0][k], 2)
	//					+ pow(hist[i][1][k] - hist[j][1][k], 2)
	//					+ pow(hist[i][2][k] - hist[j][2][k], 2);
	//			}
	//			dis = sqrt(dis);
	//			dis3x3.at<double>(i, j) = dis3x3.at<double>(j, i) = dis;
	//		}
	//	}
	//}
	//cv::Mat disSum;
	//cv::reduce(dis3x3, disSum, 0, CV_REDUCE_SUM, CV_64F);
	//for (int i = 0; i < border.size(); i++)
	//{
	//	int clusId = borderlabels.at<int>(i, 0);
	//	sumclus[clusId] += 1;
	//}
	//double* buf = disSum.ptr<double>(0);
	//for (int i = 0; i < 3; i++)
	//{
	//	*(buf + i) = (*(buf + i))*m_info.numlabels / sumclus[i];
	//}
	//int rmId = *buf > *(buf + 1) ? 0 : 1;
	//rmId = *(buf + rmId) > *(buf + 2) ? rmId : 2;
	return 0;// rmId;
}

void InitValue::getSalFromClusteredBorder(cv::Mat& unaryMap, bool illustrate)
{
	////border superpixels clustering
	//cv::Mat borderlabels;
	//std::vector<cv::Vec3f> border;
	//clusterBorder(borderlabels, border);

	////illustate the k - means result
	////for (int i = 0; i < borderIdx.size(); i++)
	////{
	////	const int clus = borderlabels.at<int>(i, 0);
	////	for (auto ite = m_info.sps[borderIdx[i]].begin(); ite < m_info.sps[borderIdx[i]].end(); ite++)
	////	{
	////		if (0 == clus) img2.at<cv::Vec3b>((*ite).y, (*ite).x) = cv::Vec3b(255, 0, 0);
	////		else if (1 == clus) img2.at<cv::Vec3b>((*ite).y, (*ite).x) = cv::Vec3b(0, 255, 0);
	////		else img2.at<cv::Vec3b>((*ite).y, (*ite).x) = cv::Vec3b(0, 0, 0);
	////	}
	////}
	////cv::imshow("original", img1);
	////cv::imshow("cluster", img2);
	////cv::waitKey(0);

	////calculate distance from each inner note to the border cluster

	////but I'd like to get the histogram of each cluster in Lab space first, in order to 
	////remove the most unlikely border cluster
	//double sumclus[3];	
	//int rmId = removeCluster(sumclus,borderlabels,border);

	////first, calculate the covariance and mean so forth.
	//Covariance clus[2];
	//std::map<int, int> ids, reids;

	//int ini(0);
	//for (int i = 0; i < 3; i++)
	//{
	//	if (i != rmId)
	//	{
	//		ids.insert(std::make_pair(i, ini));
	//		reids.insert(std::make_pair(ini, i));
	//		clus[ini].initLearning();
	//		ini++;
	//	}
	//}
	//for (int i = 0; i < border.size(); i++)
	//{
	//	int clusId = borderlabels.at<int>(i, 0);
	//	if (clusId != rmId)
	//	{
	//		clus[ids.at(clusId)].addSample(border[i]);
	//	}
	//	//sumclus[clusId] += 1; // m_info.features[borderIdx[i]][5];
	//}

	//for (int i = 0; i < 2; i++)
	//{
	//	clus[i].endLearning();
	//}

	////second, calculate distance as the initial saliency
	////std::vector<double> iniSal(innerIdx.size());
	//cv::Mat mean[2];
	//std::vector<cv::Vec2d> sal;
	//for (int i = 0; i < 2; i++)
	//{
	//	mean[i] = cv::Mat_<double>(1, 3);
	//	mean[i].at<double>(0, 0) = clus[i].mean[0];
	//	mean[i].at<double>(0, 1) = clus[i].mean[1];
	//	mean[i].at<double>(0, 2) = clus[i].mean[2];
	//}

	//for (int i = 0; i < m_info.numlabels; i++)
	//{

	//	cv::Mat lab = cv::Mat_<double>(1, 3);
	//	lab.at<double>(0, 0) = m_info.features[i][0];
	//	lab.at<double>(0, 1) = m_info.features[i][1];
	//	lab.at<double>(0, 2) = m_info.features[i][2];
	//	cv::Vec2d nodeSal;
	//	for (int j = 0; j < 2; j++)
	//	{
	//		cv::Mat re = (lab - mean[j])*(clus[j].inverseCovs)*((lab - mean[j]).t());
	//		nodeSal[j] = re.at<double>(0);
	//	}
	//	sal.push_back(nodeSal);
	//}

	////calculate unary
	//unaryMap = cv::Mat::zeros(1, m_info.numlabels, CV_32F);
	//float* unabuf = unaryMap.ptr<float>(0);
	//for (int i = 0; i < m_info.numlabels; i++)
	//{
	//	double comsal = (sal[i][0] * sumclus[reids.at(0)] +
	//		sal[i][1] * sumclus[reids.at(1)]) / (sumclus[reids.at(0)] + sumclus[reids.at(1)]);
	//	*(unabuf + i) = comsal;
	//}
	//cv::normalize(unaryMap, unaryMap, 0.0, 1.0, cv::NORM_MINMAX);

	////illustrate initial value results
	//if (illustrate)
	//{	
	//	//illustrate 3 distance maps
	//	cv::Mat map0 = cv::Mat::zeros(m_info.height, m_info.width, CV_32F);
	//	cv::Mat map1 = cv::Mat::zeros(m_info.height, m_info.width, CV_32F);
	//	//cv::Mat map2 = cv::Mat::zeros(m_info.height, m_info.width, CV_32F);
	//	cv::Mat map3 = cv::Mat::zeros(m_info.height, m_info.width, CV_32F);
	//	for (int i = 0; i < m_info.numlabels; i++)
	//	{
	//		for (auto ite = m_info.sps[i].begin(); ite < m_info.sps[i].end(); ite++)
	//		{
	//			map0.at<float>((*ite).y, (*ite).x) = sal[i][0];
	//			map1.at<float>((*ite).y, (*ite).x) = sal[i][1];
	//			//map2.at<float>((*ite).y, (*ite).x) = sal[i][2];
	//			map3.at<float>((*ite).y, (*ite).x) = *(unabuf + i);
	//		}
	//	}
	//	cv::normalize(map0, map0, 0.0, 1.0, cv::NORM_MINMAX);
	//	cv::normalize(map1, map1, 0.0, 1.0, cv::NORM_MINMAX);
	//	//cv::normalize(map2, map2, 0.0, 1.0, cv::NORM_MINMAX);

	//	//map3 = (map0 * sumclus[reids.at(0)] +
	//	//	map1 * sumclus[reids.at(1)]) / (sumclus[reids.at(0)] + sumclus[reids.at(1)]);
	//	//cv::normalize(map3, map3, 0.0, 1.0, cv::NORM_MINMAX);

	//	cv::imshow("map0", map0);
	//	cv::imshow("map1", map1);
	//	//cv::imshow("map2", map2);
	//	cv::imshow("mapcom", map3);
	//	cv::waitKey(0);
	//}
}

void InitValue::getSalFromGmmBorder(cv::Mat& unaryMap, cv::Mat& unaFuse, bool usePixel)
{
	cv::Mat segVal1f = cv::Mat::zeros(m_info.imNormLab_.size(), CV_32F);

	for (auto i : m_borderIdx)
	{
		for (auto ite = m_info.sps_[i].begin();
			ite < m_info.sps_[i].end(); ite++)
		{
			segVal1f.at<float>(ite->y, ite->x) = 1;
		}
	}

	_bGMM.BuildGMMs(m_info.imNormLab_, _bGMMidx1i, segVal1f);
	_bGMM.RefineGMMs(m_info.imNormLab_, _bGMMidx1i, segVal1f);

	const CmGaussian<3>* cmGuass = _bGMM.GetGaussians();

#pragma region illustrate cluster results
	//illustrate cluster results
	//std::vector<cv::Mat> illmap(_bGMM.K());
	//for (int i = 0; i < _bGMM.K(); i++)
	//{
	//	illmap[i] = cv::Mat::zeros(img.size(), CV_32F);
	//	for (int j = 0; j < borderIdx.size(); j++)
	//	{
	//		for (auto ite = m_info.sps[borderIdx[j]].begin(); ite < m_info.sps[borderIdx[j]].end(); ite++)
	//		{
	//			illmap[i].at<float>(ite->y, ite->x) = cmGuass[i].w * _bGMM.P(i, imgBGR3f.at<Vec3f>(ite->y, ite->x));
	//		}
	//	}
	//}

	//cv::Mat illsum = cv::Mat::zeros(img.size(), CV_32F);
	//for (int i = 0; i < _bGMM.K(); i++)
	//{
	//	cv::add(illmap[i], illsum, illsum);
	//}
	//cv::imshow("original", img);
	//for (int i = 0; i < _bGMM.K(); i++)
	//{
	//	cv::divide(illmap[i], illsum, illmap[i]);

	//	double miN, maX;
	//	cv::minMaxLoc(illmap[i], &miN, &maX);
	//	std::cout << "map " << i << " : " << miN << " to " << maX << std::endl;
	//	//cv::normalize(illmap[i], illmap[i], 0.0, 1.0, cv::NORM_MINMAX);	
	//	cv::imshow("cluster %d" + std::to_string(i), illmap[i]);
	//}
	//cv::waitKey(0);
	//return;
#pragma endregion

	if (usePixel)
	{
		unaryMap = cv::Mat::zeros(m_info.imNormLab_.size(), CV_32F);
	}
	else
	{
		unaryMap = cv::Mat::zeros(1, m_info.numlabels_, CV_32F);
	}
	
	//
	std::vector<cv::Mat> unaMap(_bGMM.K());
	for (int i = 0; i < unaMap.size(); i++) unaMap[i] = cv::Mat::zeros(unaryMap.size(), CV_32F);
	//for (int i = 0; i < unaMap.size(); i++) unaMap[i] = cv::Mat::zeros(1, m_info.numlabels_, CV_32F);
	
	double posW[2];
	double suM(0);

	if (usePixel)
	{
		for (int i = 0; i < unaryMap.rows; i++)
		{
			for (int j = 0; j < unaryMap.cols; j++)
			{
				for (int k = 0; k < _bGMM.K(); k++)
				{
					unaMap[k].at<float>(i, j) = _bGMM.P(k, m_info.imNormLab_(i, j));
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < m_info.numlabels_; i++)
		{
			for (int j = 0; j < _bGMM.K(); j++)
			{
				unaMap[j].at<float>(i) = _bGMM.P(j, m_info.features_[i].mean_normlab_);
			}
		}
	}
	

	for (int i = 0; i < unaMap.size(); i++)
	{
		cv::normalize(unaMap[i], unaMap[i], 0.0, 1.0, NORM_MINMAX);
		unaMap[i] = 1 - unaMap[i];
	}

	for (int i = 0; i < unaMap.size(); i++)
	{
		cv::add(unaMap[i] * (cmGuass[i].w), unaryMap, unaryMap);
	}
	cv::normalize(unaryMap, unaryMap, 0.0, 1.0, NORM_MINMAX);

#pragma region illustrate each map
	//illustrate each map relative to each cluster
	//std::vector<cv::Mat> illu(unaMap.size());
	//cv::Mat illtotal(img.size(), CV_32F);
	//for (int i = 0; i < unaMap.size(); i++) illu[i] = cv::Mat(img.size(), CV_32F);
	//for (int i = 0; i < m_info.numlabels; i++)
	//{
	//	for (auto ite = m_info.sps[i].begin(); ite < m_info.sps[i].end(); ite++)
	//	{
	//		//for (int j = 0; j < illu.size(); j++)
	//		//{
	//		//	illu[j].at<float>((*ite).y, (*ite).x) = unaMap[j].at<float>(i);
	//		//}
	//		illtotal.at<float>((*ite).y, (*ite).x) = unaryMap.at<float>(i);
	//	}
	//}
	//cv::imshow("original", img0);
	////for (int i = 0; i < illu.size(); i++)
	////{
	////	cv::imshow("illu" + std::to_string(i), illu[i]);
	////}
	//cv::imshow("illutotal", illtotal);
	//cv::waitKey(0);
#pragma endregion

	if (!usePixel)
	{
		cv::Mat illu(m_info.imNormLab_.size(), CV_32F);
		for (int i = 0; i < m_info.numlabels_; i++)
		{
			for (auto j : m_info.sps_[i])
			{
				illu.at<float>(j.y, j.x) = unaryMap.at<float>(i);
			}
		}
		unaryMap = illu;
	}
	
	//enhance(unaryMap);
	//fuseSpatial(unaryMap, unaFuse, pic);
	//enhance(unaFuse, 1.0);

	return;
}

void InitValue::fuseSpatial(cv::Mat& unaryMap, cv::Mat& unaFuse, const std::string& pic)
{
	CV_Assert(unaryMap.cols == m_info.numlabels_);
	const int N = unaryMap.cols;
	//float posNorm = 1.0 / max(m_info.height_, m_info.width_);

	float sigma_c_ = 1.f / 20.0;
	float sigma_p_ = 1.f / 0.25;

	////////get background weight matrix
	float sigma_b_ = 0.5;
	cv::Mat bkw = 1.0 - unaryMap;
	bkw = bkw.mul(bkw) / (-sigma_b_*sigma_b_);
	cv::exp(bkw, bkw);
	bkw = 1.f - bkw;

	std::vector< cv::Vec2f > features(N);
	cv::Mat_<float> data(N, 6);
	for (int i = 0; i < N; i++) {
		features[i] = m_info.features_[i].mean_position_ * sigma_p_;
		cv::Vec3f c = m_info.features_[i].mean_lab_;
		data(i, 0) = 1;
		data(i, 1) =  bkw.at<float>(i);
		data(i, 2) = c[0] * bkw.at<float>(i);
		data(i, 3) = c[1] * bkw.at<float>(i);
		data(i, 4) = c[2] * bkw.at<float>(i);
		data(i, 5) = c.dot(c) * bkw.at<float>(i);
		//data(i, 4) = p[0] * unaryMap.at<float>(i);
		//data(i, 5) = p[1] * unaryMap.at<float>(i);
		//data(i, 6) = unaryMap.at<float>(i);
	}

	Permutohedral* permutohedral_ = new Permutohedral();
	permutohedral_->init((const float*)features.data(), 2, N);
	permutohedral_->compute(data.ptr<float>(), data.ptr<float>(), 6, 0, 0, N, N);


	///////////
	//std::vector< Vec2f > features(stat.size());
	//Mat_<float> data(stat.size(), 5);
	//for (int i = 0; i < N; i++) {
	//	features[i] = stat[i].mean_position_ / settings_.sigma_p_;
	//	Vec3f c = stat[i].mean_color_;
	//	data(i, 0) = 1;
	//	data(i, 1) = c[0];
	//	data(i, 2) = c[1];
	//	data(i, 3) = c[2];
	//	data(i, 4) = c.dot(c);
	//}
	//// Filter
	//Filter filter((const float*)features.data(), N, 2);
	//filter.filter(data.ptr<float>(), data.ptr<float>(), 5);

	// Compute the uniqueness
	unaFuse = cv::Mat::zeros(1, N, CV_32F);
	for (int i = 0; i < N; i++) {
		cv::Vec3f c = m_info.features_[i].mean_lab_;

		unaFuse.at<float>(i) = data(i, 1)* c.dot(c)
			- 2 * (c[0] * data(i, 2) + c[1] * data(i, 3) + c[2] * data(i, 4))
			+ data(i, 5);

		//unaFuse.at<float>(i) = data(i, 0)*c.dot(c) + data(i, 4) - 2 * (c[0] * data(i, 1) + c[1] * data(i, 2) + c[2] * data(i, 3));
	}
	//normVec(r);
	/////////////////

	// Compute the uniqueness
	//std::vector< float > r(N);

	//////////////
	//std::vector< float > r(N);
	//const float sc = 0.5 / (sigma_c_*sigma_c_);
	//for (int i = 0; i < N; i++) {
	//	float u = 0, norm = 1e-10;
	//	Vec3f c = m_info.features_[i].mean_lab_;
	//	Vec2f p(0.f, 0.f);

	//	// Find the mean position
	//	for (int j = 0; j < N; j++) {
	//		Vec3f dc = m_info.features_[j].mean_lab_ - c;
	//		float w = fast_exp(-sc * dc.dot(dc));
	//		p += w*m_info.features_[j].mean_position_;
	//		norm += w;
	//	}
	//	p *= 1.0 / norm;

	//	// Compute the variance
	//	for (int j = 0; j < N; j++) {
	//		Vec3f dc = m_info.features_[j].mean_lab_ - c;
	//		Vec2f dp = m_info.features_[j].mean_position_ - p;
	//		float w = fast_exp(-sc * dc.dot(dc));
	//		u += w*dp.dot(dp);
	//	}
	//	r[i] = u / norm;
	//}

	//unaFuse = cv::Mat::zeros(1, N, CV_32F);
	//for (int i = 0; i < N; i++)
	//{
	//	//unaFuse.at<float>(i) = data(i, 1) / data(i, 0)
	//	//	- 2 * (data(i, 2)*data(i, 4) + data(i, 3)*data(i, 5)) / (std::pow(data(i, 0), 2))
	//	//	+ (data(i, 2)*data(i, 2) + data(i, 3)*data(i, 3))*data(i, 6) / std::pow(data(i, 0), 3);
	//	unaFuse.at<float>(i) = data(i, 1) / data(i, 0)
	//		- (data(i, 2)*data(i, 2) + data(i, 3)*data(i, 3)) / (data(i, 0) * data(i, 0));
	//	//unaFuse.at<float>(i) = r[i];
	//}
	//r[i] = data(i, 3) / data(i, 0) - (data(i, 1) * data(i, 1) + data(i, 2) * data(i, 2)) / (data(i, 0) * data(i, 0));

	//normVec(r);
	cv::normalize(unaFuse, unaFuse, 0.0, 1.0, NORM_MINMAX);
	//cv::exp(unaFuse*(-3.0), unaFuse);
	//cv::normalize(unaFuse, unaFuse, 0.0, 1.0, NORM_MINMAX);
	return;
}

void InitValue::enhance(cv::Mat& unaryMap, double fct)
{
	//CV_Assert(unaryMap.cols == m_info.numlabels_);
	cv::Mat thresMap;
	//double s = 0;
	//for (int i = 0; i < unaryMap.cols; i++)
	//{
	//	s += m_info.features_[i].size_ * unaryMap.at<float>(i);
	//}
	//double imMean = fct * s / (m_info.height_*m_info.width_);
	double imMean = fct * cv::mean(unaryMap)[0];

	cv::exp((unaryMap - imMean)*(-20.0), thresMap);
	thresMap = 1.0 / (1.0 + thresMap);
	cv::normalize(thresMap, unaryMap, 0.0, 1.0, NORM_MINMAX);
}

void InitValue::morphSmooth(const cv::Mat& dMap, cv::Mat& dst)
{
	double smooth_alpha = 50;
	int radius = floor(smooth_alpha*std::sqrt(cv::mean(dMap).val[0]));
	radius = radius > 3 ? radius : 3;

	cv::Mat img;
	dMap.convertTo(img, CV_8UC1, 255);
	//cv::imshow("img", img);
	cv::Mat ker = getStructuringElement(cv::MORPH_RECT, cv::Size(radius, radius));
	cv::Mat Ie;
	cv::erode(img, Ie, ker);
	cv::Mat Iobr(img.size(), img.type()), Iobrd;
	cv::Mat Iobrcbr(img.size(), img.type());

	imreconstruct(img.data, Ie.data, 4, img.rows, img.cols, Iobr.data);

	cv::dilate(Iobr, Iobrd, ker);
	Iobr = 255 - Iobr; Iobrd = 255 - Iobrd;

	imreconstruct(Iobr.data, Iobrd.data, 4, img.rows, img.cols, Iobrcbr.data);

	Iobrcbr = 255 - Iobrcbr;
	Iobrcbr.convertTo(Iobrcbr, CV_32FC1);
	cv::normalize(Iobrcbr, dst, 0.0, 1.0, cv::NORM_MINMAX);
	//Iobrcbr.copyTo(dMap);
	return;
}

cv::Mat InitValue::boxfilter(const cv::Mat& imSrc, int r)
{
	assert(imSrc.channels() == 1);
	int hei = imSrc.rows;
	int wid = imSrc.cols;

	cv::Mat imDst = cv::Mat::zeros(imSrc.size(), CV_32F);

	//cumulative sum over Y axis
	//cv::Mat tmat = imSrc.t();
	cv::Mat imCum = imSrc.t();

	for (int j = 0; j < imCum.rows; j++)
	{
		//const unsigned char* tmatbuf = tmat.ptr<unsigned char>(j);
		float* cumbuf = imCum.ptr<float>(j);
		for (int i = 1; i < imCum.cols; i++)
		{
			*(cumbuf + i) += *(cumbuf + i - 1);
		}
	}
	imCum = imCum.t();

	//difference over Y axis
	imCum.rowRange(r, 2 * r + 1).copyTo(imDst.rowRange(0, r + 1));
	cv::Mat tmat = imCum.rowRange(2 * r + 1, hei) - imCum.rowRange(0, hei - 2 * r - 1);
	tmat.copyTo(imDst.rowRange(r + 1, hei - r));

	tmat = cv::repeat(imCum.row(hei - 1), r, 1) - imCum.rowRange(hei - 2 * r - 1, hei - r - 1);
	tmat.copyTo(imDst.rowRange(hei - r, hei));

	//cumulative sum over X axis
	imCum = imDst.clone();
	for (int j = 0; j < imCum.rows; j++)
	{
		float* cumbuf = imCum.ptr<float>(j);
		for (int i = 1; i < imCum.cols; i++)
		{
			*(cumbuf + i) += *(cumbuf + i - 1);
		}
	}

	//difference over Y axis
	imCum.colRange(r, 2 * r + 1).copyTo(imDst.colRange(0, r + 1));
	tmat = imCum.colRange(2 * r + 1, wid) - imCum.colRange(0, wid - 2 * r - 1);
	tmat.copyTo(imDst.colRange(r + 1, wid - r));
	tmat = cv::repeat(imCum.col(wid - 1), 1, r) - imCum.colRange(wid - 2 * r - 1, wid - r - 1);
	tmat.copyTo(imDst.colRange(wid - r, wid));

	return imDst;
}

cv::Mat InitValue::fastguidedfilter(const cv::Mat& I, const cv::Mat& p, int r, double eps, int s)
{
	cv::Mat I_sub, P_sub;
	double fct = 1.0 / s;
	cv::resize(I, I_sub, cv::Size(0,0), fct, fct, INTER_NEAREST);
	cv::resize(p, P_sub, cv::Size(0, 0), fct, fct, INTER_NEAREST);

	int r_sub = r / s;

	int hei = I_sub.rows;
	int wid = I_sub.cols;

	cv::Mat N = boxfilter(cv::Mat::ones(hei, wid, CV_32F), r_sub);


	cv::Mat mean_I = boxfilter(I_sub, r_sub) / N;
	cv::Mat mean_P = boxfilter(P_sub, r_sub) / N;
	cv::Mat mean_Ip = boxfilter(I_sub.mul(P_sub), r_sub) / N;
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_P);

	

	cv::Mat mean_II = boxfilter(I_sub.mul(I_sub), r_sub) / N;
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	cv::Mat a = cov_Ip / (var_I + eps);
	cv::Mat b = mean_P - a.mul(mean_I);

	cv::Mat mean_a = boxfilter(a, r_sub) / N;
	cv::Mat mean_b = boxfilter(b, r_sub) / N;

	cv::resize(mean_a, mean_a, I.size(), 0, 0, INTER_LINEAR);
	cv::resize(mean_b, mean_b, I.size(), 0, 0, INTER_LINEAR);

	cv::Mat tmat;
	I.convertTo(tmat, CV_32F);
	cv::Mat q = mean_a.mul(tmat) + mean_b;

	return q;
}

void InitValue::enhanceWithGuidedFilter(const cv::Mat& I, cv::Mat& dst)
{
	assert(I.type() == CV_8UC3);

	cv::Mat dI;
	I.convertTo(dI, CV_32F, 1/255.0);
	std::vector<cv::Mat> mv;
	cv::split(dI, mv);

	int r = 16;
	double eps = 0.01;
	int s = 4;

	for (int i = 0; i < mv.size(); i++)
	{
		mv[i] = fastguidedfilter(mv[i], mv[i], r, eps, s);
	}

	cv::merge(mv, dst);
	dst.convertTo(dst, CV_8U, 255.0);
}

bool InitValue::removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi)
{
	assert(inImg.type() == CV_8UC3);
	if (inImg.rows < 2 * (FRAME_MAX + 3) || inImg.cols < 2 * (FRAME_MAX + 3))
	{
		roi = cv::Rect(0, 0, inImg.cols, inImg.rows);
		outImg = inImg;
		return false;
	}

	cv::Mat imgGray;
	cvtColor(inImg, imgGray, CV_BGR2GRAY);

	int up, dn, lf, rt;

	up = findFrameMargin(imgGray.rowRange(0, FRAME_MAX), false);
	dn = findFrameMargin(imgGray.rowRange(imgGray.rows - FRAME_MAX, imgGray.rows), true);
	lf = findFrameMargin(imgGray.colRange(0, FRAME_MAX).t(), false);
	rt = findFrameMargin(imgGray.colRange(imgGray.cols - FRAME_MAX, imgGray.cols).t(), true);

	int margin = MAX(up, MAX(dn, MAX(lf, rt)));
	if (margin == 0)
	{
		roi = cv::Rect(0, 0, imgGray.cols, imgGray.rows);
		outImg = inImg;
		return false;
	}

	int count = 0;
	count = up == 0 ? count : count + 1;
	count = dn == 0 ? count : count + 1;
	count = lf == 0 ? count : count + 1;
	count = rt == 0 ? count : count + 1;

	// cut four border region if at least 2 border frames are detected
	if (count > 1)
	{
		margin += 2;
		roi = cv::Rect(margin, margin, inImg.cols - 2 * margin, inImg.rows - 2 * margin);
		outImg = cv::Mat(inImg, roi);

		return true;
	}

	// otherwise, cut only one border
	up = up == 0 ? up : up + 2;
	dn = dn == 0 ? dn : dn + 2;
	lf = lf == 0 ? lf : lf + 2;
	rt = rt == 0 ? rt : rt + 2;


	roi = cv::Rect(lf, up, inImg.cols - lf - rt, inImg.rows - up - dn);
	outImg = cv::Mat(inImg, roi);

	return true;
}

int InitValue::findFrameMargin(const cv::Mat& img, bool reverse)
{
	cv::Mat edgeMap, edgeMapDil, edgeMask;
	Sobel(img, edgeMap, CV_16SC1, 0, 1);
	edgeMap = abs(edgeMap);
	edgeMap.convertTo(edgeMap, CV_8UC1);
	edgeMask = edgeMap < (SOBEL_THRESH * 255.0);
	dilate(edgeMap, edgeMapDil, cv::Mat(), cv::Point(-1, -1), 2);
	edgeMap = edgeMap == edgeMapDil;
	edgeMap.setTo(Scalar(0.0), edgeMask);


	if (!reverse)
	{
		for (int i = edgeMap.rows - 1; i >= 0; i--)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return i + 1;
	}
	else
	{
		for (int i = 0; i < edgeMap.rows; i++)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return edgeMap.rows - i;
	}

	return 0;
}