//Author: Zhuo Su, in Beihang University (BUAA)
//date: 04/2017

#pragma once

#include"InitValue.h"

class Automata
{
public:
	void work(const cv::Mat& src, InitValue& initVal, cv::Mat& dst);

private:
	cv::Mat_<float> getImpactM(InitValue& initVal, cv::Mat& rmsk);
	cv::Mat_<float> getCoherenceM(const cv::Mat_<float>& impactM);
	void getInitSal(const cv::Mat& src, InitValue& iniVal, cv::Mat_<float>& initSal);

	void inference(const InitValue& initVal, const cv::Mat_<float>& F_normal,
		const cv::Mat_<float>& C_normal, cv::Mat_<float>& initSal, const cv::Mat& msk);
};