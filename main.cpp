
#include <vector>
#include <string>
#include <iostream>

#include "PictureHandler.h"
#include "SLIC.h"
#include"InitValue.h"
#include"FineValue.h"
#include"Automata.h"

#include<opencv2/opencv.hpp>

void beshowable(cv::Mat& illmor, std::string jpgname = "", bool wr = false)
{
	illmor.convertTo(illmor, CV_8U, 255.0);

	if (wr)
	{
		cv::imwrite(jpgname, illmor);
	}
	
	std::vector<cv::Mat> illmors(3);
	for (int i = 0; i < 3; i++) illmors[i] = illmor;
	cv::merge(illmors, illmor);
}

//using namespace Gdiplus;
//using namespace std;
int main4()//4
{
	cv::Mat_<float> ma = cv::Mat_<float>::zeros(4, 4);
	cv::Mat re;
	ma(0, 2) = 1;
	ma(2, 2) = 3;
	std::cout << ma << std::endl;
	re = cv::Mat::ones(4, 1, CV_32F) * 2;
	re = cv::Mat::diag(re);
	std::cout << std::endl << re << std::endl;
	ma = re*ma;
	std::cout << std::endl << ma;
	//cv::reduce(ma, re, 1, CV_REDUCE_SUM);
	//re = ma > 0;
	//re /= 255;
	//int s = cv::sum(re)[0];
	//auto tp = re.type();
	//std::cout << "num: " << s << std::endl;
	std::cin.get();
	return 0;
}

int main3()//3
{
	//string filebase = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\Imgs\\77.jpg";
	string filebase = "E:\\lab\\C_C++\\semantic-segmentation\\salient\\images\\0021.jpg";
	cv::Mat img = cv::imread(filebase);
	cv::Mat fstimg;
	cv::Rect rect;
	bool isrm = InitValue::removeFrame(img, fstimg, rect);
	//InitValue::enhanceWithGuidedFilter(img, fstimg);
	//cv::Mat imga;
	//cv::cvtColor(img, imga, CV_BGR2BGRA);
	//std::vector<cv::Mat> mv;
	//cv::split(imga, mv);
	//mv.pop_back();
	//cv::Mat a;
	//cv::merge(mv, a);
	cv::imshow("original", img);
	//cv::imshow("a", a);
	cv::imshow("fstimg", fstimg);
	cv::waitKey(0);
	return 0;
}

int main2()//2
{
	while (1)
	{
		std::string jpgname;
		std::cout << "\nplease input jpg number" << std::endl;
		std::cin >> jpgname;
		if (jpgname == "q") break;
		//std::string jpgname = "137";
		string filebase = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\guided_filter\\";
		//string filebase = "E:\\lab\\C_C++\\semantic-segmentation\\salient\\images\\";
		string pic = filebase + jpgname + ".jpg";
		string picout = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\hisgoEqua2\\" + jpgname + ".jpg";

		cv::Mat img = cv::imread(pic);
		cv::Mat outImg;
		std::vector<cv::Mat> mv, outMv(3);
		cv::split(img, mv);
		for (int i = 0; i < 3; i++)
		{
			cv::equalizeHist(mv[i], outMv[i]);
		}
		cv::merge(outMv, outImg);
		bool res = cv::imwrite(picout, outImg);
		std::cout << "finished "<<res<<" : " << jpgname;
		//cv::imshow("original", img);
		//cv::imshow("after histogram equalization", outImg);
		//cv::waitKey(0);
	}
	return 0;

}

int main()//1
{
	while (1){
	InitValue initval;
	//_w1, _w2, _w3, _alpha, _beta, _gama, _mu
	//appearance: w1 alpha beta , combine color(alpha) and location(beta)
	//location smoothness: w2 gama
	//color similarity: w3 mu	
	//FineValue fineval(0, 0, 3, 40, 63, 6, 43, 2);//6,10,2,20,33,3,43
	//FineValue fineval(4/*w1*/, 0/*w2*/, 0/*w3*/, 40/*color(alpha)*/,
	//	33/*location(beta)*/, 2/*smoothness*/, 40/*similarity*/, 4);

	//good
	//FineValue fineval(2.5/*w1*/, 0.5/*w2*/, 0/*w3*/, 0.3/*color(alpha)*/,
	//	0.2/*location(beta)*/, 0.08/*smoothness*/, 0.1/*similarity*/, 2);

	FineValue fineval(2.5/*w1*/, 0.5/*w2*/, 0/*w3*/, 0.3/*color(alpha)*/,
		0.2/*location(beta)*/, 0.08/*smoothness*/, 0.1/*similarity*/, 2);
	//FineValue fineval(10/*w1*/, 3/*w2*/, 3/*w3*/, 80/*color(alpha)*/,
	//	30/*location(beta)*/, 30/*smoothness*/, 4/*similarity*/, 4);
	cv::Mat unaryMap, unaFuse;
	std::string jpgname;
	std::cout << "\nplease input jpg number" << std::endl;
	std::cin >> jpgname;
	if (jpgname == "q") break;
	//std::string jpgname = "137";
	//string filebase = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\Imgs\\";
	string filebase = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\prob\\";
	//string filebase = "E:\\lab\\C_C++\\semantic-segmentation\\salient\\images\\";
	string pic = filebase + jpgname + ".jpg";
	cv::Mat img = cv::imread(pic);
	//cv::Mat img;
	//InitValue::enhanceWithGuidedFilter(img_ori, img);
	cv::Mat imgrm;
	cv::Rect rect;
	bool rm = InitValue::removeFrame(img, imgrm, rect);
	//bool rm = false;
	initval.GetBgvalue(unaryMap, unaFuse, imgrm, false);


	cv::Mat illmor;
	InitValue::morphSmooth(unaryMap,illmor);

	cv::Mat fineMap;
	//fineval.getFineVal(initval, illmor, fineMap);
	Automata automata;
	automata.work(illmor, initval, fineMap);
	
	if (rm)
	{
		cv::Mat tmat;
		tmat = cv::Mat::zeros(img.size(), CV_32F);
		illmor.copyTo(tmat(rect));
		illmor = tmat;
		tmat = cv::Mat::zeros(img.size(), CV_32F);
		fineMap.copyTo(tmat(rect));
		fineMap = tmat;
	}
	//InitValue::morphSmooth(fineMap, fineMap);

	
	cv::Mat IMGSHOW = cv::Mat::zeros(2 * img.rows + 5, 2 * img.cols + 5, CV_8UC3);
	img.copyTo(IMGSHOW(cv::Rect(0, 0, img.cols, img.rows)));
	
	//img.copyTo(IMGSHOW(cv::Rect(img.cols + 4, 0, img.cols, img.rows)));


	beshowable(illmor);
	illmor.copyTo(IMGSHOW(cv::Rect(0, img.rows + 4, illmor.cols, illmor.rows)));

	beshowable(fineMap);
	fineMap.copyTo(IMGSHOW(cv::Rect(img.cols + 4, img.rows + 4, fineMap.cols, fineMap.rows)));

	cv::imshow("original adn results", IMGSHOW);
	//cv::imshow("unary Map", unaryMap);
	//cv::imshow("mor map", illmor);
	//cv::imshow("unary2 map", illUna2);
	//cv::imshow("final map", fineMap);
	cv::waitKey(0);
	}
	return 0;

}

int main0()//0
{
	PictureHandler picHand;
	vector<string> picvec(0);
	picvec.resize(0);
	//GetPictures(picvec);//user chooses one or more pictures
	string pic = "..\\..\\MSRA10K_Imgs_GT\\Imgs\\101.jpg";
	string saveLocation = "..\\..\\result\\";
	//BrowseForFolder(saveLocation);

	int numPics(picvec.size());

	int m_spcount = 200;
	double m_compactness = 20.0;

	for (int k = 0; k < 1; k++)
	{
		unsigned int* img = nullptr;
		int width(0);
		int height(0);

		picHand.GetPictureBuffer(pic, img, width, height);


		cv::Mat im(height, width, CV_8UC3);
		int i, j, idx;
		unsigned char* buf = im.data;
		unsigned char* imgbuf = (unsigned char*)(img);
		for (i = 0; i < height*width; i++)
		{
			*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 0));//b
			*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 1));//g
			*(buf++) = (unsigned char)(*(imgbuf + i * 4 + 2));//r
		}
		cv::imshow("hehe", im);
		cv::waitKey(0);
		return 0;

		int sz = width*height;
		//---------------------------------------------------------
		if (m_spcount < 20 || m_spcount > sz / 4) m_spcount = sz / 200;//i.e the default size of the superpixel is 200 pixels
		if (m_compactness < 1.0 || m_compactness > 80.0) m_compactness = 20.0;
		//---------------------------------------------------------
		int* labels = new int[sz];
		int numlabels(0);
		SLIC slic;
		slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, width, height, labels, numlabels, m_spcount, m_compactness);
		//slic.DoSuperpixelSegmentation_ForGivenSuperpixelSize(img, width, height, labels, numlabels, 10, m_compactness);//demo
		//slic.DrawContoursAroundSegments(img, labels, width, height, 0);
		
		if (labels) delete[] labels;

		//picHand.SavePicture(img, width, height, pic, saveLocation, 1, "_SLIC");// 0 is for BMP and 1 for JPEG)
		if (img) delete[] img;
	}
	std::cout << "finished" << std::endl;
	std::cin.get();
	return 0;
}