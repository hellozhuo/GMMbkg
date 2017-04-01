#pragma once

#include"InfoRetrieval.h"
#include"CmGMM.h"

class InitValue
{
public:
	InfoRetrieval m_info;

	std::vector<int> borderIdx, innerIdx;

	CmGMM _bGMM, _fGMM; // Background and foreground GMM
	Mat _bGMMidx1i, _fGMMidx1i;	// Background and foreground GMM components, supply memory for GMM, not used for Grabcut 

public:
	InitValue()
		:_bGMM(4), _fGMM(4){}
	void GetBgvalue(cv::Mat& unaryMap, const std::string& pic);

	void getIdxs();

	void clusterBorder(cv::Mat& borderlabels, std::vector<cv::Vec3f>& border);
	int removeCluster(double sumclus[3], cv::Mat& borderlabels, std::vector<cv::Vec3f>& border);
	void getSalFromClusteredBorder(cv::Mat& unaryMap, bool illustrate = false);

	void getSalFromGmmBorder(cv::Mat& unaryMap, const std::string& pic);
};

class Covariance
{
public:
	cv::Mat cov;

	cv::Mat inverseCovs;
	//double inverseCovs[componentsCount][3][3]; //协方差的逆矩阵  
	double mean[3];
	double covDeterms;  //协方差的行列式  

	double sums[3];
	double prods[3][3];
	int sampleCounts;
	//int totalSampleCount;

public:
	void initLearning()
	{
		sums[0] = sums[1] = sums[2] = 0;
		prods[0][0] = prods[0][1] = prods[0][2] = 0;
		prods[1][0] = prods[1][1] = prods[1][2] = 0;
		prods[2][0] = prods[2][1] = prods[2][2] = 0;
		sampleCounts = 0;
	}

	void addSample(const cv::Vec3d color)
	{
		sums[0] += color[0]; sums[1] += color[1]; sums[2] += color[2];
		prods[0][0] += color[0] * color[0]; prods[0][1] += color[0] * color[1]; prods[0][2] += color[0] * color[2];
		prods[1][0] += color[1] * color[0]; prods[1][1] += color[1] * color[1]; prods[1][2] += color[1] * color[2];
		prods[2][0] += color[2] * color[0]; prods[2][1] += color[2] * color[1]; prods[2][2] += color[2] * color[2];
		sampleCounts++;
	}

	void calcInverseCovAndDeterm(double* c)
	{
		double dtrm =
			covDeterms = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6])
			+ c[2] * (c[3] * c[7] - c[4] * c[6]);

		//在C++中，每一种内置的数据类型都拥有不同的属性, 使用<limits>库可以获  
		//得这些基本数据类型的数值属性。因为浮点算法的截断，所以使得，当a=2，  
		//b=3时 10*a/b == 20/b不成立。那怎么办呢？  
		//这个小正数（epsilon）常量就来了，小正数通常为可用给定数据类型的  
		//大于1的最小值与1之差来表示。若dtrm结果不大于小正数，那么它几乎为零。  
		//所以下式保证dtrm>0，即行列式的计算正确（协方差对称正定，故行列式大于0）。  
		CV_Assert(dtrm > std::numeric_limits<double>::epsilon());
		//三阶方阵的求逆  
		inverseCovs = cv::Mat_<double>(3, 3);
		inverseCovs.at<double>(0,0) = (c[4] * c[8] - c[5] * c[7]) / dtrm;
		inverseCovs.at<double>(1, 0) = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
		inverseCovs.at<double>(2, 0) = (c[3] * c[7] - c[4] * c[6]) / dtrm;
		inverseCovs.at<double>(0, 1) = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
		inverseCovs.at<double>(1, 1) = (c[0] * c[8] - c[2] * c[6]) / dtrm;
		inverseCovs.at<double>(2, 1) = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
		inverseCovs.at<double>(0, 2) = (c[1] * c[5] - c[2] * c[4]) / dtrm;
		inverseCovs.at<double>(1, 2) = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
		inverseCovs.at<double>(2, 2) = (c[0] * c[4] - c[1] * c[3]) / dtrm;
	}

	void endLearning()
	{
		int n = sampleCounts;
		const double variance = 0.01;
		mean[0] = sums[0] / n; mean[1] = sums[1] / n; mean[2] = sums[2] / n;

		//计算第ci个高斯模型的协方差  
		cov = cv::Mat_<double>(3, 3);
		double c[9];
		
		c[0] = cov.at<double>(0, 0) = prods[0][0] / n - mean[0] * mean[0];
		c[1] = cov.at<double>(0, 1) = prods[0][1] / n - mean[0] * mean[1];
		c[2] = cov.at<double>(0, 2) = prods[0][2] / n - mean[0] * mean[2];
		c[3] = cov.at<double>(1, 0) = prods[1][0] / n - mean[1] * mean[0];
		c[4] = cov.at<double>(1, 1) = prods[1][1] / n - mean[1] * mean[1];
		c[5] = cov.at<double>(1, 2) = prods[1][2] / n - mean[1] * mean[2];
		c[6] = cov.at<double>(2, 0) = prods[2][0] / n - mean[2] * mean[0];
		c[7] = cov.at<double>(2, 1) = prods[2][1] / n - mean[2] * mean[1];
		c[8] = cov.at<double>(2, 2) = prods[2][2] / n - mean[2] * mean[2];

		//计算第ci个高斯模型的协方差的行列式  
		double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
		if (dtrm <= std::numeric_limits<double>::epsilon())
		{
			//相当于如果行列式小于等于0，（对角线元素）增加白噪声，避免其变  
			//为退化（降秩）协方差矩阵（不存在逆矩阵，但后面的计算需要计算逆矩阵）。  
			// Adds the white noise to avoid singular covariance matrix.
			c[0] += variance;
			c[4] += variance;
			c[8] += variance;
			cov.at<double>(0, 0) += variance;
			cov.at<double>(1, 1) += variance;
			cov.at<double>(2, 2) += variance;
		}

		//计算第ci个高斯模型的协方差的逆Inverse和行列式Determinant  
		calcInverseCovAndDeterm(c);
	}

};