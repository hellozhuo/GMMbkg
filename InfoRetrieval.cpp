#include"InfoRetrieval.h"

SuperpixelInfo::SuperpixelInfo()
	:mean_lab_(0.f, 0.f, 0.f), mean_normlab_(0.f,0.f,0.f), mean_bgr_(0.f, 0.f, 0.f), mean_position_(0.f, 0.f), isborder_(false)
{}

void InfoRetrieval::GetInfomation(std::string filename, int spcount, double compactness)
{
	unsigned int* img = nullptr;
	int* labelsbuf = nullptr;
	picHand_.GetPictureBuffer(filename, img, width_, height_);
	int sz = width_*height_;
	if (spcount < 20 || spcount > sz / 4) spcount = sz / 200;//i.e the default size of the superpixel is 200 pixels
	if (compactness < 1.0 || compactness > 80.0) compactness = 20.0;
	//---------------------------------------------------------
	//labels = make_unique_array<int>(sz);

	SLIC slic;
	//auto labelsbuf = labels.get();
	
	DestroyFeatures();
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img,
		width_, height_, labelsbuf, numlabels_, spcount, compactness);

	RetrieveOnSP(img,labelsbuf);
	if (img) delete[] img;
	if (labelsbuf) delete[] labelsbuf;
}

//[L a b sx sy pixelNumber isborder] 7 dimensions
void InfoRetrieval::RetrieveOnSP(const unsigned int* img,const int* labelsbuf)
{
	int i, j, idx;
	sps_.resize(numlabels_);
	features_.resize(numlabels_);
	for (i = 0; i < numlabels_; i++)
	{
		sps_[i].clear();
	}

	cv::Mat_<cv::Vec3b> im(height_, width_);
	unsigned char* buf = im.data;
	const unsigned char* imgbuf = (const unsigned char*)(img);
	for (i = 0; i < height_*width_; i++)
	{
		*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 0));//b
		*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 1));//g
		*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 2));//r
	}
	cv::Mat imLab3b;
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
			idx = *(labelsbuf++);
			sps_[idx].push_back(cv::Point(j, i));
			features_[idx].mean_lab_ += imLab_(i, j);
			features_[idx].mean_normlab_ += imNormLab_(i, j);
			features_[idx].mean_bgr_ += im(i, j);
			features_[idx].mean_position_ += cv::Vec2f(j, i);
			tm_count[idx]++;
			if (i == 0 || i == height_ - 1 || j == 0 || j == width_ - 1) features_[idx].isborder_ = true;
		}
	}
	for (i = 0; i < numlabels_; i++)
	{
		features_[i].mean_lab_ *= 1.0 / tm_count[i];
		features_[i].mean_normlab_ *= 1.0 / tm_count[i];
		features_[i].mean_bgr_ *= 1.0 / tm_count[i];
		features_[i].mean_position_ *= 1.0 / tm_count[i];
		features_[i].size_ = tm_count[i];
	}
	for (i = 0; i < numlabels_; i++)
	{
		features_[i].mean_position_ *= 1.0 / (max(height_, width_));
	}
	return;
}

void InfoRetrieval::DestroyFeatures()
{
	//if (features&&numlabels > 0)
	//{
	//	for (int i = 0; i < numlabels;i++)
	//	{
	//		delete[] features[i];
	//	}
	//	delete[] features;
	//	features = nullptr;
	//}
}