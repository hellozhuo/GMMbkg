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
#include"fastmath.h"
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

struct SuperpixelInfo {
	cv::Vec3f mean_lab_;//L [0 100] a [-127 127] b [-127 127]
	cv::Vec3f mean_normlab_;//[0 1]
	cv::Vec3f mean_bgr_;//[0 255]
	cv::Vec2f mean_position_;//[0 1]
	std::set<int, std::less<int>> neighbor_;
	std::set<int, std::less<int>> neighborCnt_;
	bool isborder_;
	int size_;
	SuperpixelInfo();
};

class InfoRetrieval
{
public:
	//unsigned int* img; //data of original img
	//std::unique_ptr<int[]> labels; //label of each pixel
	//std::unique_ptr<unsigned int[]> border; //[L a b sx sy pixelNumber] of border superpixels
	//std::unique_ptr<unsigned int[]> inner; //[L a b sx sy pixelNumber] of inner superpixels
	std::vector<SuperpixelInfo> features_;
	std::vector<std::vector<cv::Point>> sps_;
	//cv::Mat inputImg;
	int height_, width_, numlabels_;
	int sz_;
	cv::Mat_<cv::Vec3f> imLab_;//normal Lab range
	cv::Mat_<cv::Vec3f> imNormLab_;//[0 1]
	cv::Mat_<cv::Vec3b> imBgr_;//[0 255]
	//PictureHandler picHand_;
	int* labelsbuf_;
	bool nb_;
	bool nbCnt_;//record if nods share the same borders with the neighbor nodes are calculated

public:
	InfoRetrieval()
		:height_(0), width_(0), numlabels_(0), labelsbuf_(nullptr), nb_(false), nbCnt_(false){}
	~InfoRetrieval()
	{
		//DestroyFeatures();
		//if (img) delete[] img;
		if (labelsbuf_)
		{
			delete[] labelsbuf_;
			labelsbuf_ = nullptr;
		}
	}
	void GetInfomation(const cv::Mat& im, int spcount, double compactness);
	void getNeighbor(const int* const labelsbuf);
	void getNeighborCnt();

private:
	//get information to border and inner
	void RetrieveOnSP(const cv::Mat_<cv::Vec3b>& im, const int* const labelsbuf);
	//void DestroyFeatures();
};