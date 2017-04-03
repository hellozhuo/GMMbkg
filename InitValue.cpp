#include"InitValue.h"
#include"permutohedral.h"
#include<map>
#include<algorithm>
#include<functional>

void InitValue::GetBgvalue(cv::Mat& unaryMap, cv::Mat& unaFuse, const std::string& pic)
{
	//string pic = "..\\..\\MSRA10K_Imgs_GT\\Imgs\\938.jpg";
	int spcount = 300;
	double compactness = 20.0;
	this->m_info.GetInfomation(pic, spcount, compactness);

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
	getSalFromGmmBorder(unaryMap, unaFuse, pic);

	return;
}

void InitValue::getIdxs()
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
	
	m_borderIdx2.clear();
	for (auto i : m_borderIdx)
	{
		for (auto j : m_info.features_[i].neighbor_)
		{
			m_borderIdx2.insert(j);
		}
	}
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

void InitValue::getSalFromGmmBorder(cv::Mat& unaryMap, cv::Mat& unaFuse, const std::string& pic)
{
	//cv::Mat img0 = cv::imread(pic);
	//cv::Mat img;
	//cv::cvtColor(img0, img, CV_BGR2Lab);
	cv::Mat segVal1f = cv::Mat::zeros(m_info.imNormLab_.size(), CV_32F);
	//cv::Mat imgBGR3f;
	//img.convertTo(imgBGR3f, CV_32FC3, 1 / 255.0);
	//for (int i = 0; i < m_borderIdx.size(); i++)
	for (auto i:m_borderIdx)
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

	unaryMap = cv::Mat::zeros(1, m_info.numlabels_, CV_32F);
	std::vector<cv::Mat> unaMap(_bGMM.K());
	for (int i = 0; i < unaMap.size(); i++) unaMap[i] = cv::Mat::zeros(1, m_info.numlabels_, CV_32F);
	
	//std::map<double, int, std::greater<double>> dnIds;
	//for (int i = 0; i < _bGMM.K(); i++)
	//{
	//	dnIds.insert(std::make_pair(cmGuass[i].w, i));
	//}
	double posW[2];
	double suM(0);
	//auto ite = dnIds.begin();
	//for (int i = 0; i < 3; i++, ite++) suM += cmGuass[ite->second].w;
	//ite = dnIds.begin();
	//for (int i = 0; i < 2; i++, ite++) posW[i] = cmGuass[ite->second].w;// / suM;

	//float dot[3];
	//float* unabuf = unaryMap.ptr<float>(0);
	for (int i = 0; i < m_info.numlabels_; i++)
	{
		//dot[0] = m_info.features[i][0] / 255.0;//B or L
		//dot[1] = m_info.features[i][1] / 255.0;//G or a
		//dot[2] = m_info.features[i][2] / 255.0;//R or b
		

		//suM = 0;
		//ite = dnIds.begin();
		//for (int j = 0; j < 2; j++, ite++)
		//{
		//	suM += posW[j] * _bGMM.P(ite->second, dot);
		//}

		//*(unabuf + i) = 1-suM;
		for (int j = 0; j < _bGMM.K(); j++)
		{
			unaMap[j].at<float>(i) = _bGMM.P(j, m_info.features_[i].mean_normlab_);
		}
	}

	for (int i = 0; i < unaMap.size(); i++)
	{
		//unaMap[i] = 1 - unaMap[i];
		cv::normalize(unaMap[i], unaMap[i], 0.0, 1.0, NORM_MINMAX);
		unaMap[i] = 1 - unaMap[i];
	}

	//unatotal = cv::Mat::zeros(1, m_info.numlabels, CV_32F);
	for (int i = 0; i < unaMap.size(); i++)
	{
		cv::add(unaMap[i] * (cmGuass[i].w), unaryMap, unaryMap);
	}
	cv::normalize(unaryMap, unaryMap, 0.0, 1.0, NORM_MINMAX);
	//cv::normalize(unaryMap, unaryMap, 0.0, 1.0, NORM_MINMAX);
	//unaryMap = 1 - unaryMap;

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
	enhance(unaryMap);
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
	CV_Assert(unaryMap.cols == m_info.numlabels_);
	cv::Mat thresMap;
	double s = 0;
	for (int i = 0; i < unaryMap.cols; i++)
	{
		s += m_info.features_[i].size_ * unaryMap.at<float>(i);
	}
	double imMean = fct * s / (m_info.height_*m_info.width_);

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