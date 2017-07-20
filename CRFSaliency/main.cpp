//Author: Zhuo Su, in Beihang University (BUAA)
//date: 04/2017



#include <vector>
#include <string>
#include <iostream>
#include<fstream>

#include "PictureHandler.h"
#include "SLIC.h"
#include"InitValue.h"
#include"FineValue.h"
#include"Automata.h"
#include"direct.h"
#include<opencv2/opencv.hpp>

#include<chrono>
using namespace std::chrono;

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

std::vector<std::string> fileList(std::string dirroot, std::string ext)
{
	WIN32_FIND_DATAA fileFindData;
	std::string nameW = dirroot + ext;
	std::string filename;
	int fileindex(0);
	std::vector<std::string> jpglist;
	jpglist.reserve(1000);
	HANDLE hFind = ::FindFirstFileA(nameW.c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		return jpglist;
	}
	do{
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue; // Ignore sub-folders

		filename = fileFindData.cFileName;
		jpglist.push_back(filename);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return jpglist;
}


int main()//1
{
	std::string filebase = "..\\imgs\\";
	//std::string fileload = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\yanshi\\";
	//std::string fileload2 = "..\\..\\ECSSD_AUTO700\\";
	//_mkdir(fileload.c_str());
	//_mkdir(fileload2.c_str());
	//std::string record = "E:\\lab\\C_C++\\saliency-detection\\DUTOMRON_RESULTS\\DUTOMRON_GMMbkg_t.txt";
	//std::vector<std::string> jpglist = fileList(filebase, "*.jpg");
	//const int N = jpglist.size();
	//assert(N > 1);
	//time_point<high_resolution_clock> m_begin;
	//m_begin = high_resolution_clock::now();
	int autocount_ = 0;
	//for (int jpgid = 0; jpgid < 1; jpgid++)
	while (1)
	{
		//bool useauto = false;
	InitValue initval;
	//_w1, _w2, _w3, _alpha, _beta, _gama, _mu
	//appearance: w1 alpha beta , combine color(alpha) and location(beta)
	//location smoothness: w2 gama
	//color similarity: w3 mu	
	//FineValue fineval(0, 0, 3, 40, 63, 6, 43, 2);//6,10,2,20,33,3,43
	//FineValue fineval(4/*w1*/, 0/*w2*/, 0/*w3*/, 40/*color(alpha)*/,
	//	33/*location(beta)*/, 2/*smoothness*/, 40/*similarity*/, 4);

	//good
	FineValue fineval(2.5/*w1*/, 0.5/*w2*/, 0/*w3*/, 0.3/*color(alpha)*/,
		0.2/*location(beta)*/, 0.08/*smoothness*/, 0.1/*similarity*/, 2);

	//FineValue fineval(2.5/*w1*/, 0.5/*w2*/, 0/*w3*/, 0.3/*color(alpha)*/,
	//	0.2/*location(beta)*/, 0.08/*smoothness*/, 0.1/*similarity*/, 2);
	//FineValue fineval(10/*w1*/, 3/*w2*/, 3/*w3*/, 80/*color(alpha)*/,
	//	30/*location(beta)*/, 30/*smoothness*/, 4/*similarity*/, 4);
	cv::Mat unaryMap, unaFuse;
	//std::string jpgname;
	//std::cout << "\nplease input jpg number" << std::endl;
	//std::cin >> jpgname;
	//if (jpgname == "q") break;
	//std::string jpgname = "137";
	//string filebase = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\Imgs\\";
	//string filebase = "E:\\lab\\C_C++\\saliency-detection\\code\\SuZhuo\\MSRA10K_Imgs_GT\\prob\\";
	
	//string pic = filebase + jpgname + ".jpg";
	//std::string pic = filebase + jpglist[jpgid];
	std::string jname;
	std::cin >> jname;
	if (jname == "q") break;

	//std::string picload = fileload + jpglist[jpgid].substr(0, jpglist[jpgid].length() - 4) + ".png";
	std::string pic = filebase + jname + ".jpg";
	//std::string picload2 = fileload2 + jpglist[jpgid].substr(0, jpglist[jpgid].length() - 4) + ".png";
	cv::Mat img = cv::imread(pic);
	//cv::Mat img;
	//InitValue::enhanceWithGuidedFilter(img_ori, img);
	cv::Mat imgrm;
	cv::Rect rect;
	bool rm = InitValue::removeFrame(img, imgrm, rect);
	initval.GetBgvalue(unaryMap, unaFuse, imgrm, false);

	//cv::imshow("before enhance", unaryMap);
	//cv::waitKey(0);
	unaFuse = unaryMap.clone();

	initval.enhance(unaryMap);
	//cv::imshow("after enhance", unaryMap);
	//cv::waitKey(0);

	cv::Mat illmor;
	InitValue::morphSmooth(unaryMap,illmor);

	//cv::imshow("illmor", illmor);
	//cv::waitKey(0);

	cv::Mat fineMap;
	fineval.getFineVal(initval, illmor, fineMap);

	Automata automata;
	double ithres = cv::mean(fineMap)[0];
	cv::Mat upsal = fineMap > 0.5;
	float nUp = cv::sum(upsal / 255)[0];
	float rt = 0.045;
	if (nUp <= rt*fineMap.size().area())
	{
		//std::cout << "\tautomata" << std::endl;
		autocount_++;
		//useauto = true;
		automata.work(illmor, initval, fineMap);
	}	

	if (rm)
	{
		//cv::Mat tmat;
		//tmat = cv::Mat::zeros(img.size(), CV_32F);
		//illmor.copyTo(tmat(rect));
		//illmor = tmat;	
		cv::Mat tmat1;
		tmat1 = cv::Mat::zeros(img.size(), CV_32F);
		fineMap.copyTo(tmat1(rect));
		fineMap = tmat1;
	}
	
	fineMap.convertTo(fineMap, CV_8U, 255);
	//unaryMap.convertTo(unaryMap, CV_8U, 255);
	//unaFuse.convertTo(unaFuse, CV_8U, 255);
	//illmor.convertTo(illmor, CV_8U, 255);

	//imwrite(fileload + jname + "_before.png", unaFuse);
	//imwrite(fileload + jname + "_after.png", unaryMap);
	//imwrite(fileload + jname + "_illmor.png", illmor);
	//imwrite(fileload + jname + "_GMMbkg.png", fineMap);
	cv::imshow("GMMbkg", fineMap);
	cv::waitKey(0);
	//std::cout << "finish " << jname << std::endl;
	//cv::Mat yanshi = cv::Mat::zeros(unaryMap.rows, 3 * unaryMap.cols + 10, CV_8U);
	//unaFuse.copyTo(yanshi(cv::Rect(0, 0, unaFuse.cols, unaFuse.rows)));
	//unaryMap.copyTo(yanshi(cv::Rect(unaFuse.cols + 4, 0, unaryMap.cols, unaryMap.rows)));
	//illmor.copyTo(yanshi(cv::Rect(2 * unaryMap.cols + 8, 0, illmor.cols, illmor.rows)));
	//cv::imshow("yanshi", yanshi);
	//cv::waitKey(0);
	//cv::imwrite(picload, fineMap);
	
	//cv::Mat IMGSHOW = cv::Mat::zeros(2 * img.rows + 5, 2 * img.cols + 5, CV_8UC3);
	//img.copyTo(IMGSHOW(cv::Rect(0, 0, img.cols, img.rows)));

	//beshowable(illmor);
	//illmor.copyTo(IMGSHOW(cv::Rect(0, img.rows + 4, illmor.cols, illmor.rows)));

	//beshowable(fineMap);
	//fineMap.copyTo(IMGSHOW(cv::Rect(img.cols + 4, img.rows + 4, fineMap.cols, fineMap.rows)));

	//cv::imshow("original adn results", IMGSHOW);
	//cv::waitKey(0);
	}
	//int64_t dur = duration_cast<milliseconds>(high_resolution_clock::now() - m_begin).count();
	//int64_t avedur = dur / jpglist.size();
	//std::cout << "time : " << dur << " milliseconds" << std::endl;
	//std::cout << "finished with average time: "<< avedur <<" milliseconds" << std::endl;
	//std::ofstream fil(record);
	//fil << "automata : " << autocount_ << " times" << std::endl;
	//fil << "finished with average time: " << avedur << " milliseconds" << std::endl;
	//fil.close();
	std::cin.get();

	return 0;

}

int main0()//0
{
	PictureHandler picHand;
	vector<string> picvec(0);
	picvec.resize(0);
	//GetPictures(picvec);//user chooses one or more pictures
	string pic = "..\\..\\ecssd_images\\0004.jpg";
	string saveLocation = "..\\..\\superpixel\\";
	//BrowseForFolder(saveLocation);
	_mkdir(saveLocation.c_str());

	int numPics(picvec.size());

	int m_spcount = 300;
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
		slic.DrawContoursAroundSegments(img, labels, width, height, 0);
		
		if (labels) delete[] labels;

		picHand.SavePicture(img, width, height, pic, saveLocation, 1, "_SLIC");// 0 is for BMP and 1 for JPEG)
		if (img) delete[] img;
	}
	std::cout << "finished" << std::endl;
	std::cin.get();
	return 0;
}