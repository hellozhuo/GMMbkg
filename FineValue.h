#pragma once

#include"InitValue.h"

#include "DenseCRF.h"

class FineValue
{
public:
	cv::Mat _resLabels;
	float _w1, _w2, _w3, _alpha, _beta, _gama, _mu;
	int _iter;

public:
	FineValue(float w1, float w2, float w3, float alpha,
		float beta, float gama, float mu,int ite)
		:_w1(w1), _w2(w2), _w3(w3), _alpha(alpha), _beta(beta),_gama(gama), _mu(mu),_iter(ite){

	}

	void getFineVal(const InitValue& initVal, const cv::Mat& unaryMap, cv::Mat& fineMap);

private:
	float * classify(const InitValue& initVal, const cv::Mat& note);

	void proceed(const InitValue& initVal, float* unary, int iter = 4);
};