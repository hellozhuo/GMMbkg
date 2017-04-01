#include"InfoRetrieval.h"

void InfoRetrieval::GetInfomation(std::string filename, int spcount, double compactness)
{
	unsigned int* img = nullptr;
	int* labelsbuf = nullptr;
	picHand.GetPictureBuffer(filename, img, width, height);
	int sz = width*height;
	if (spcount < 20 || spcount > sz / 4) spcount = sz / 200;//i.e the default size of the superpixel is 200 pixels
	if (compactness < 1.0 || compactness > 80.0) compactness = 20.0;
	//---------------------------------------------------------
	//labels = make_unique_array<int>(sz);

	SLIC slic;
	//auto labelsbuf = labels.get();
	
	DestroyFeatures();
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img,
		width, height, labelsbuf, numlabels, spcount, compactness);

	RetrieveOnSP(img,labelsbuf);
	if (img) delete[] img;
	if (labelsbuf) delete[] labelsbuf;
}

//[L a b sx sy pixelNumber isborder] 7 dimensions
void InfoRetrieval::RetrieveOnSP(unsigned int* img, int* labelsbuf)
{
	int i, j, idx;
	sps.resize(numlabels);
	features = new unsigned int*[numlabels];
	for (i = 0; i < numlabels; i++)
	{
		sps[i].clear();
		features[i] = new unsigned int[7];
		memset(features[i], 0, 7 * sizeof(unsigned int));
	}
	//memset(features, 0, numlabels * 7 * sizeof(unsigned int));
	
	inputImg = cv::Mat(height, width, CV_8UC3);
	const cv::Mat& im = inputImg;
	unsigned char* buf = im.data;
	unsigned char* imgbuf = (unsigned char*)(img);
	for (i = 0; i < height*width; i++)
	{
		*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 0));//b
		*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 1));//g
		*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 2));//r
	}
	cv::cvtColor(im, im, CV_BGR2Lab);

	//auto labelbuf = labels.get();
	buf = im.data;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			idx = *(labelsbuf++);
			if (idx >= numlabels)
			{
				int a = 0;
			}
			sps[idx].push_back(cv::Point(j, i));
			features[idx][0] += *(buf++);//L or B
			features[idx][1] += *(buf++);//a or G
			features[idx][2] += *(buf++);//b or R
			features[idx][3] += j;//sx
			features[idx][4] += i;//sy
			features[idx][5]++;//number of pixels
			if (i == 0 || i == height - 1 || j == 0 || j == width - 1) features[idx][6] = 1;
		}
	}
	for (i = 0; i < numlabels; i++)
	{
		for (j = 0; j < 5; j++)
		{
			features[i][j] /= features[i][5];
		}	
	}
}

void InfoRetrieval::DestroyFeatures()
{
	if (features&&numlabels > 0)
	{
		for (int i = 0; i < numlabels;i++)
		{
			delete[] features[i];
		}
		delete[] features;
		features = nullptr;
	}
}