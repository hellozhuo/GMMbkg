
#include <vector>
#include <string>
#include <iostream>

#include "PictureHandler.h"
#include "SLIC.h"
#include"InitValue.h"
#include"FineValue.h"

#include<opencv2/opencv.hpp>

//using namespace Gdiplus;
//using namespace std;
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
	FineValue fineval(0, 0, 3, 40, 63, 6, 43, 2);//6,10,2,20,33,3,43
	cv::Mat unaryMap, unaFuse;
	std::string jpgname;
	std::cout << "\nplease input jpg number" << std::endl;
	std::cin >> jpgname;
	if (jpgname == "q") break;
	//std::string jpgname = "137";
	string filebase = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\guided_filter\\";
	//string filebase2 = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\prob\\";
	//string filebase = "E:\\lab\\C_C++\\semantic-segmentation\\salient\\images\\";
	string pic = filebase + jpgname + ".jpg";
	//string pic2 = filebase2 + jpgname + ".jpg";
	//string pic = "..\\..\\MSRA10K_Imgs_GT\\Imgs\\" + jpgname + ".jpg";
	initval.GetBgvalue(unaryMap, unaFuse, pic);
	//initval2.GetBgvalue(unaryMap2, pic2);
	//for (int i = 0; i < initval.m_info.numlabels; i++)
	//{
	//	unaFuse.at<float>(i) = exp(-settings_.k_ * dist[i]);
	//}
	cv::Mat unafinal;
	cv::exp(unaFuse*3.0, unafinal);
	cv::multiply(unaryMap, unafinal, unafinal);
	cv::normalize(unafinal, unafinal, 0.0, 1.0, NORM_MINMAX);

	//continue;

	//////////////////////////////////////////////////////////////////////////
	//double uMean = cv::mean(unaryMap)[0];
	////double uMin, uMax;
	////cv::minMaxLoc(unaryMap, &uMin, &uMax);

	//double a1 = uMean;// *2.0 / 3.0;//b1=1.0
	//double b2 = 1 - uMean / 5.0;
	//double a2 = (a1 + 1 - uMean / 3.0) / 2.0;
	//double tp = (b2 - a2) / (1 - uMean);
	//cv::Mat una2;
	//cv::Mat msk = unaryMap > a1;
	//cv::Mat msk2 = unaryMap <= a1;
	//cv::subtract(unaryMap,a1,una2,msk);
	//cv::subtract(una2, una2, una2, msk2);
	//una2 = una2*tp;
	//cv::add(una2, a2, una2, msk);
	//cv::add(una2, unaryMap, una2, msk2);


	//(unaryMap - uMean) / (1 - uMean)*(b2 - a2) + a2;

	//illustrate unary map
	cv::Mat img = cv::imread(pic);
	cv::Mat illUnary(img.size(), CV_32F);
	cv::Mat illUnary2(img.size(), CV_32F);
	cv::Mat illUnaryfinal(img.size(), CV_32F);
	for (int i = 0; i < initval.m_info.numlabels_; i++)
	{
		for (auto ite = initval.m_info.sps_[i].begin(); ite < initval.m_info.sps_[i].end(); ite++)
		{
			illUnary.at<float>((*ite).y, (*ite).x) = unaFuse.at<float>(i);
			illUnary2.at<float>((*ite).y, (*ite).x) = unaryMap.at<float>(i);
			illUnaryfinal.at<float>((*ite).y, (*ite).x) = unafinal.at<float>(i);
		}
	}

	//double m_sal = 0.1 * img.cols*img.rows;
	//for (float sm = sum(illUnary)[0]; sm < m_sal; sm = sum(illUnary)[0])
	//	illUnary = min(illUnary*m_sal / sm, 1.0f);

	////cv::Mat imr;
	////illUnary.convertTo(imr, CV_8U,255.0);
	////cv::imshow("unary map", illUnary);
	////cv::waitKey(0);

	////bool res = cv::imwrite("..\\..\\result\\"+jpgname+".png", imr);
	////std::cout << "finished "<<res << std::endl;
	////std::cin.get();
	////return 0;

	//cv::Mat thresMap;
	//double imMean = 1.8 * cv::mean(illUnary)[0];

	//cv::exp((illUnary - imMean)*(-20.0), thresMap);
	//thresMap = 1.0 / (1.0 + thresMap);
	//cv::normalize(thresMap, thresMap, 0.0, 1.0, NORM_MINMAX);

	//illUnary.convertTo(thresMap, CV_8U, 255.0);
	//cv::threshold(illUnary, thresMap, imMean, 1.0, CV_THRESH_BINARY);
	cv::imshow("original", img);	
	cv::imshow("spatial map", illUnary);
	cv::imshow("unary map", illUnary2);
	cv::imshow("unary final", illUnaryfinal);
	cv::waitKey(0);
		//break;
	}
	return 0;

	//fineval.getFineVal(initval,unaryMap);
	////fineval.getFineVal(initval, una2);

	////illustrate final map
	//cv::Mat illFinal(img.size(), CV_32F);
	//for (int i = 0; i < initval.m_info.numlabels; i++)
	//{
	//	for (auto ite = initval.m_info.sps[i].begin(); ite < initval.m_info.sps[i].end(); ite++)
	//	{

	//		illFinal.at<float>((*ite).y, (*ite).x) = fineval.resLabels.at<float>(i);
	//	}
	//}
	//cv::imshow("original", img);
	//cv::imshow("unary map", illUnary);
	////cv::imshow("unary2 map", illUna2);
	//cv::imshow("final map", illFinal);
	//cv::waitKey(0);

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