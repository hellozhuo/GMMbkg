#pragma once

#include<memory>
#include<string>
#include<vector>
#include<array>

#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "PictureHandler.h"
#include "SLIC.h"

template<typename T, size_t size, typename... Args>
std::unique_ptr<T[]> make_unique_array(Args... args)
{
	return std::unique_ptr<T[], std::default_delete<T[]>>(new T[size]{args...});
}

template<typename T>
std::unique_ptr<T[]> make_unique_array(size_t size)
{
	return std::unique_ptr<T[], std::default_delete<T[]>>(new T[size]);
}

class InfoRetrieval
{
public:
	//unsigned int* img; //data of original img
	//std::unique_ptr<int[]> labels; //label of each pixel
	//std::unique_ptr<unsigned int[]> border; //[L a b sx sy pixelNumber] of border superpixels
	//std::unique_ptr<unsigned int[]> inner; //[L a b sx sy pixelNumber] of inner superpixels
	unsigned int** features;
	std::vector<std::vector<cv::Point>> sps;
	cv::Mat inputImg;
	int height, width, numlabels;
	PictureHandler picHand;

public:
	InfoRetrieval()
		:height(0), width(0), numlabels(0){}
	~InfoRetrieval()
	{
		DestroyFeatures();
		//if (img) delete[] img;
	}
	void GetInfomation(std::string filename, int spcount, double compactness);

private:
	//get information to border and inner
	void RetrieveOnSP(unsigned int* img, int* labels);
	void DestroyFeatures();
};