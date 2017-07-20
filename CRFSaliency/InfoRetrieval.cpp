//Author: Zhuo Su, in Beihang University (BUAA)
//date: 04/2017


#include"InfoRetrieval.h"
#include<fstream>

SuperpixelInfo::SuperpixelInfo()
	:mean_lab_(0.f, 0.f, 0.f), mean_normlab_(0.f,0.f,0.f), /*mean_bgr_(0.f, 0.f, 0.f),*/ mean_position_(0.f, 0.f), isborder_(false)
{}

void InfoRetrieval::GetInfomation(const cv::Mat& im, int spcount, double compactness)
{
	assert(im.type() == CV_8UC3);
	cv::Mat imga;
	cv::cvtColor(im, imga, CV_BGR2BGRA);
	width_ = im.cols;
	height_ = im.rows;
	sz_ = width_*height_;
	unsigned int* img = (unsigned int*)imga.data;
	int* labelsbuf = nullptr;
	//picHand_.GetPictureBuffer(filename, img, width_, height_);

	int sz = width_*height_;
	if (spcount < 20 || spcount > sz / 4) spcount = sz / 200;//i.e the default size of the superpixel is 200 pixels
	if (compactness < 1.0 || compactness > 80.0) compactness = 20.0;
	//---------------------------------------------------------
	//labels = make_unique_array<int>(sz);

	SLIC slic;
	//auto labelsbuf = labels.get();
	
	//DestroyFeatures();
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img,
		width_, height_, labelsbuf, numlabels_, spcount, compactness);

	//std::ifstream fil("E:\\lab\\C_C++\\saliency-detection\\code\\automata\\superpixels\\0042.dat", std::ios_base::binary);
	//labelsbuf = new int[sz];
	//fil.read((char*)labelsbuf, sz*sizeof(int));
	//numlabels_ = 0;
	//for (int i = 0; i < sz; i++)
	//{
	//	int a = *(labelsbuf + i);
	//	if (a > numlabels_) numlabels_ = a;
	//}
	//numlabels_++;
	//fil.close();
	labelsbuf_ = labelsbuf;
	labelsbuf = nullptr;

	RetrieveOnSP(im,labelsbuf_);
}

//[L a b sx sy pixelNumber isborder] 7 dimensions
void InfoRetrieval::RetrieveOnSP(const cv::Mat_<cv::Vec3b>& im, const int* const labelsbuf)
{
	int i, j, idx;
	sps_.resize(numlabels_);
	features_.resize(numlabels_);
	for (i = 0; i < numlabels_; i++)
	{
		sps_[i].clear();
		features_[i].neighbor_.clear();
		features_[i].neighborCnt_.clear();
	}

	cv::Mat imLab3b;
	//imBgr_ = im.clone();
	cv::cvtColor(im, imLab3b, CV_BGR2Lab);
	imLab3b.convertTo(imNormLab_, CV_32F, 1 / 255.0);

	cv::Mat_<cv::Vec3f> imLab(im.size());
	im.convertTo(imLab, CV_32F, 1 / 255.0);

	//L [0 100] a [-127 127] b [-127 127]
	cv::cvtColor(imLab, imLab_, CV_BGR2Lab);

	std::vector<double> tm_count(numlabels_, 1e-10);
	for (i = 0; i < height_; i++)
	{
		for (j = 0; j < width_; j++)
		{
			idx = *(labelsbuf + i*width_ + j);
			sps_[idx].push_back(cv::Point(j, i));
			features_[idx].mean_lab_ += imLab_(i, j);
			features_[idx].mean_normlab_ += imNormLab_(i, j);
			//features_[idx].mean_bgr_ += im(i, j);
			features_[idx].mean_position_ += cv::Vec2f(j, i);//x,y
			tm_count[idx]++;
			if (i == 0 || i == height_ - 1 || j == 0 || j == width_ - 1) features_[idx].isborder_ = true;
		}
	}
	for (i = 0; i < numlabels_; i++)
	{
		features_[i].mean_lab_ *= 1.0 / tm_count[i];
		features_[i].mean_normlab_ *= 1.0 / tm_count[i];
		//features_[i].mean_bgr_ *= 1.0 / tm_count[i];
		features_[i].mean_position_ *= 1.0 / tm_count[i];
		features_[i].size_ = tm_count[i];
	}
	for (i = 0; i < numlabels_; i++)
	{
		features_[i].mean_position_ *= 1.0 / (max(height_, width_));
	}
	nb_ = nbCnt_ = false;
	return;
}

void InfoRetrieval::getNeighbor(const int* const labelsbuf)
{
	if (nb_) return;
	int i, j, idx1, idx2;
	for (j = 0; j < height_ - 1; j++)
	{
		for (i = 0; i < width_ - 1; i++)
		{
			idx1 = *(labelsbuf + j*width_ + i);
			idx2 = *(labelsbuf + j*width_ + i + 1);
			if (idx1 != idx2)
			{
				features_[idx1].neighbor_.insert(idx2);
				features_[idx2].neighbor_.insert(idx1);
			}
			idx2 = *(labelsbuf + (j + 1)*width_ + i + 1);
			if (idx1 != idx2)
			{
				features_[idx1].neighbor_.insert(idx2);
				features_[idx2].neighbor_.insert(idx1);
			}
			idx2 = *(labelsbuf + (j + 1)*width_ + i);
			if (idx1 != idx2)
			{
				features_[idx1].neighbor_.insert(idx2);
				features_[idx2].neighbor_.insert(idx1);
			}

			idx1 = *(labelsbuf + j*width_ + i + 1);
			idx2 = *(labelsbuf + (j + 1)*width_ + i);
			if (idx1 != idx2)
			{
				features_[idx1].neighbor_.insert(idx2);
				features_[idx2].neighbor_.insert(idx1);
			}
		}
	}
	nb_ = true;
}

void InfoRetrieval::getNeighborCnt()
{
	//if (nbCnt_) return;
	//getNeighbor(labelsbuf_);

	//for (int i = 0; i < numlabels_; i++)
	//{
	//	features_[i].neighborCnt_.clear();
	//	for (auto j : features_[i].neighbor_)
	//	{
	//		features_[i].neighborCnt_.insert(j);
	//	}
	//}

	//nbCnt_ = true;
}

