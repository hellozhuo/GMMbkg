#include"InitValue.h"
#include<map>
#include<algorithm>
#include<functional>

void InitValue::GetBgvalue(cv::Mat& unaryMap,const std::string& pic)
{
	//string pic = "..\\..\\MSRA10K_Imgs_GT\\Imgs\\938.jpg";
	int spcount = 200;
	double compactness = 20.0;
	this->m_info.GetInfomation(pic, spcount, compactness);

#pragma region illustrate the border
	//illustate the border
	//cv::Mat img1 = cv::imread(pic);
	//cv::Mat img2 = img1.clone();
	//for (int i = 0; i < m_info.numlabels; i++)
	//{
	//	if (m_info.features[i][6])
	//	{
	//		for (auto bg = m_info.sps[i].begin(); bg < m_info.sps[i].end(); bg++)
	//		{
	//			for (int j = 0; j < 3; j++)
	//				img.at<cv::Vec3b>((*bg).y, (*bg).x)[j] = i;
	//		}
	//	}
	//}
	//cv::imshow("border", img);
	//cv::waitKey(0);
#pragma endregion

	getIdxs();

	//getSalFromClusteredBorder(unaryMap);
	getSalFromGmmBorder(unaryMap,pic);

	return;
}

void InitValue::getIdxs()
{
	if (borderIdx.size() > 0 || innerIdx.size() > 0) return;

	for (int i = 0; i < m_info.numlabels; i++)
	{
		if (m_info.features[i][6])
		{
			borderIdx.push_back(i);
		}
		else
		{
			innerIdx.push_back(i);
		}
	}
}

void InitValue::clusterBorder(cv::Mat& borderlabels, std::vector<cv::Vec3f>& border)
{
	for (int i = 0; i < m_info.numlabels; i++)
	{
		if (m_info.features[i][6])
		{
			border.push_back(cv::Vec3f(m_info.features[i][0],
				m_info.features[i][1], m_info.features[i][2]));
		}
	}

	CV_Assert(!border.empty());
	cv::Mat _bgdSamples((int)border.size(), 3, CV_32FC1, &border[0][0]);
	kmeans(_bgdSamples, 3, borderlabels,
		cv::TermCriteria(CV_TERMCRIT_ITER, 10, 0.0), 0, cv::KMEANS_PP_CENTERS);
}

int InitValue::removeCluster(double sumclus[3], cv::Mat& borderlabels, std::vector<cv::Vec3f>& border)
{
	std::array<std::array<int, 256>, 3> hist[3];
	memset(sumclus, 0, 3*sizeof(double));//sizeof(sumclus)=8,not 24
	//double sumclus[3] = { 0.0, 0.0, 0.0 };
	for (int i = 0; i < 3; i++)
	{
		for (auto ite = hist[i].begin(); ite < hist[i].end(); ite++)
			ite->assign(0);
	}
	const cv::Mat& img = m_info.inputImg;
	for (int i = 0; i < borderIdx.size(); i++)
	{
		const int clus = borderlabels.at<int>(i, 0);
		for (auto ite = m_info.sps[borderIdx[i]].begin(); ite < m_info.sps[borderIdx[i]].end(); ite++)
		{
			for (int j = 0; j < 3; j++)//L a b color
			{
				hist[clus][j][img.at<cv::Vec3b>((*ite).y, (*ite).x)[j]]++;
			}
		}
	}

	//and then, calculate the inter-distance between each cluster pair
	//this code may be accelerated
	cv::Mat dis3x3 = cv::Mat_<double>(3, 3);
	for (int i = 0; i < 3; i++)
	{
		for (int j = i; j < 3; j++)
		{
			if (i == j) dis3x3.at<double>(i, j) = 0;
			else
			{
				double dis = 0.0;
				for (int k = 0; k < 256; k++)
				{
					dis += pow(hist[i][0][k] - hist[j][0][k], 2)
						+ pow(hist[i][1][k] - hist[j][1][k], 2)
						+ pow(hist[i][2][k] - hist[j][2][k], 2);
				}
				dis = sqrt(dis);
				dis3x3.at<double>(i, j) = dis3x3.at<double>(j, i) = dis;
			}
		}
	}
	cv::Mat disSum;
	cv::reduce(dis3x3, disSum, 0, CV_REDUCE_SUM, CV_64F);
	for (int i = 0; i < border.size(); i++)
	{
		int clusId = borderlabels.at<int>(i, 0);
		sumclus[clusId] += 1;
	}
	double* buf = disSum.ptr<double>(0);
	for (int i = 0; i < 3; i++)
	{
		*(buf + i) = (*(buf + i))*m_info.numlabels / sumclus[i];
	}
	int rmId = *buf > *(buf + 1) ? 0 : 1;
	rmId = *(buf + rmId) > *(buf + 2) ? rmId : 2;
	return rmId;
}

void InitValue::getSalFromClusteredBorder(cv::Mat& unaryMap, bool illustrate)
{
	//border superpixels clustering
	cv::Mat borderlabels;
	std::vector<cv::Vec3f> border;
	clusterBorder(borderlabels, border);

	//illustate the k - means result
	//for (int i = 0; i < borderIdx.size(); i++)
	//{
	//	const int clus = borderlabels.at<int>(i, 0);
	//	for (auto ite = m_info.sps[borderIdx[i]].begin(); ite < m_info.sps[borderIdx[i]].end(); ite++)
	//	{
	//		if (0 == clus) img2.at<cv::Vec3b>((*ite).y, (*ite).x) = cv::Vec3b(255, 0, 0);
	//		else if (1 == clus) img2.at<cv::Vec3b>((*ite).y, (*ite).x) = cv::Vec3b(0, 255, 0);
	//		else img2.at<cv::Vec3b>((*ite).y, (*ite).x) = cv::Vec3b(0, 0, 0);
	//	}
	//}
	//cv::imshow("original", img1);
	//cv::imshow("cluster", img2);
	//cv::waitKey(0);

	//calculate distance from each inner note to the border cluster

	//but I'd like to get the histogram of each cluster in Lab space first, in order to 
	//remove the most unlikely border cluster
	double sumclus[3];	
	int rmId = removeCluster(sumclus,borderlabels,border);

	//first, calculate the covariance and mean so forth.
	Covariance clus[2];
	std::map<int, int> ids, reids;

	int ini(0);
	for (int i = 0; i < 3; i++)
	{
		if (i != rmId)
		{
			ids.insert(std::make_pair(i, ini));
			reids.insert(std::make_pair(ini, i));
			clus[ini].initLearning();
			ini++;
		}
	}
	for (int i = 0; i < border.size(); i++)
	{
		int clusId = borderlabels.at<int>(i, 0);
		if (clusId != rmId)
		{
			clus[ids.at(clusId)].addSample(border[i]);
		}
		//sumclus[clusId] += 1; // m_info.features[borderIdx[i]][5];
	}

	for (int i = 0; i < 2; i++)
	{
		clus[i].endLearning();
	}

	//second, calculate distance as the initial saliency
	//std::vector<double> iniSal(innerIdx.size());
	cv::Mat mean[2];
	std::vector<cv::Vec2d> sal;
	for (int i = 0; i < 2; i++)
	{
		mean[i] = cv::Mat_<double>(1, 3);
		mean[i].at<double>(0, 0) = clus[i].mean[0];
		mean[i].at<double>(0, 1) = clus[i].mean[1];
		mean[i].at<double>(0, 2) = clus[i].mean[2];
	}

	for (int i = 0; i < m_info.numlabels; i++)
	{

		cv::Mat lab = cv::Mat_<double>(1, 3);
		lab.at<double>(0, 0) = m_info.features[i][0];
		lab.at<double>(0, 1) = m_info.features[i][1];
		lab.at<double>(0, 2) = m_info.features[i][2];
		cv::Vec2d nodeSal;
		for (int j = 0; j < 2; j++)
		{
			cv::Mat re = (lab - mean[j])*(clus[j].inverseCovs)*((lab - mean[j]).t());
			nodeSal[j] = re.at<double>(0);
		}
		sal.push_back(nodeSal);
	}

	//calculate unary
	unaryMap = cv::Mat::zeros(1, m_info.numlabels, CV_32F);
	float* unabuf = unaryMap.ptr<float>(0);
	for (int i = 0; i < m_info.numlabels; i++)
	{
		double comsal = (sal[i][0] * sumclus[reids.at(0)] +
			sal[i][1] * sumclus[reids.at(1)]) / (sumclus[reids.at(0)] + sumclus[reids.at(1)]);
		*(unabuf + i) = comsal;
	}
	cv::normalize(unaryMap, unaryMap, 0.0, 1.0, cv::NORM_MINMAX);

	//illustrate initial value results
	if (illustrate)
	{	
		//illustrate 3 distance maps
		cv::Mat map0 = cv::Mat::zeros(m_info.height, m_info.width, CV_32F);
		cv::Mat map1 = cv::Mat::zeros(m_info.height, m_info.width, CV_32F);
		//cv::Mat map2 = cv::Mat::zeros(m_info.height, m_info.width, CV_32F);
		cv::Mat map3 = cv::Mat::zeros(m_info.height, m_info.width, CV_32F);
		for (int i = 0; i < m_info.numlabels; i++)
		{
			for (auto ite = m_info.sps[i].begin(); ite < m_info.sps[i].end(); ite++)
			{
				map0.at<float>((*ite).y, (*ite).x) = sal[i][0];
				map1.at<float>((*ite).y, (*ite).x) = sal[i][1];
				//map2.at<float>((*ite).y, (*ite).x) = sal[i][2];
				map3.at<float>((*ite).y, (*ite).x) = *(unabuf + i);
			}
		}
		cv::normalize(map0, map0, 0.0, 1.0, cv::NORM_MINMAX);
		cv::normalize(map1, map1, 0.0, 1.0, cv::NORM_MINMAX);
		//cv::normalize(map2, map2, 0.0, 1.0, cv::NORM_MINMAX);

		//map3 = (map0 * sumclus[reids.at(0)] +
		//	map1 * sumclus[reids.at(1)]) / (sumclus[reids.at(0)] + sumclus[reids.at(1)]);
		//cv::normalize(map3, map3, 0.0, 1.0, cv::NORM_MINMAX);

		cv::imshow("map0", map0);
		cv::imshow("map1", map1);
		//cv::imshow("map2", map2);
		cv::imshow("mapcom", map3);
		cv::waitKey(0);
	}
}

void InitValue::getSalFromGmmBorder(cv::Mat& unaryMap, const std::string& pic)
{
	cv::Mat img = cv::imread(pic);
	cv::Mat segVal1f = cv::Mat::zeros(img.size(), CV_32F);
	cv::Mat imgBGR3f;
	img.convertTo(imgBGR3f, CV_32FC3, 1 / 255.0);
	for (int i = 0; i < borderIdx.size(); i++)
	{
		for (auto ite = m_info.sps[borderIdx[i]].begin();
			ite < m_info.sps[borderIdx[i]].end(); ite++)
		{
			segVal1f.at<float>(ite->y, ite->x) = 1;
		}
	}
	_bGMM.BuildGMMs(imgBGR3f, _bGMMidx1i, segVal1f);
	_bGMM.RefineGMMs(imgBGR3f, _bGMMidx1i, segVal1f);

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

	unaryMap = cv::Mat::zeros(1, m_info.numlabels, CV_32F);
	
	std::map<double, int, std::greater<double>> dnIds;
	for (int i = 0; i < _bGMM.K(); i++)
	{
		dnIds.insert(std::make_pair(cmGuass[i].w, i));
	}
	double posW[3];
	double suM(0);
	auto ite = dnIds.begin();
	for (int i = 0; i < 3; i++, ite++) suM += cmGuass[ite->second].w;
	ite = dnIds.begin();
	for (int i = 0; i < 3; i++, ite++) posW[i] = cmGuass[ite->second].w / suM;

	float dot[3];
	float* unabuf = unaryMap.ptr<float>(0);
	for (int i = 0; i < m_info.numlabels; i++)
	{
		dot[0] = m_info.features[i][0] / 255.0;//B or L
		dot[1] = m_info.features[i][1] / 255.0;//G or a
		dot[2] = m_info.features[i][2] / 255.0;//R or b

		suM = 0;
		ite = dnIds.begin();
		for (int j = 0; j < 3; j++, ite++)
		{
			suM += posW[j] * _bGMM.P(ite->second, dot);
		}

		*(unabuf + i) = suM;
	}
	cv::normalize(unaryMap, unaryMap, 0.0, 1.0, NORM_MINMAX);
	unaryMap = 1 - unaryMap;
	return;
}