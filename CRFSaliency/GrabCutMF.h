#pragma once
#include "DenseCRF.h"
#include"CmGMM.h"
#include"InitValue.h"


class GrabCutMF
{
public: // Functions for saliency cut
	// User supplied Trimap values
	//enum TrimapValue {UserBack = 0, ProbBack = 64, TrimapUnknown = 128, ProbFore = 192, UserFore = 255};

	GrabCutMF(bool usePixel, const InitValue& initVal, float w1, float w2, float w3, float alpha, float beta, float gama, float mu);

public: // Functions for GrabCut
	//Mat setGrabReg(const Rect &rect, CMat &bordMask1u);

	// Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper 
	void initialize(float* unary);

	//Mat initialize(const Mat &map1f, float backW = 10.f);

	// Run Grabcut refinement on the hard segmentation
	void refine(int iter = 4);

	// Edit Trimap, mask values should be 0 or 255
	//void setTrimap(CMat &mask1u, const TrimapValue t) {_trimap1i.setTo(t, mask1u);}

	// Get Trimap for effective interaction. Format: CV_32SC1. Values should be TrimapValue
	//Mat& getTrimap() {return _trimap1i; }

	// Draw result
	//Mat drawResult();

	//Mat showMedialResults(CStr& title);
	
	//static void Demo(CStr &wkDir, float w1, float w2, float w3, float alpha, float beta, float gama, float mu);
	//static void Demo2(CMat& src, CMat& mask, Mat& dst1, Mat& dst2, float w1, float w2, float w3, float alpha, float beta, float gama, float mu);
	//static void Demo_ori(CStr &wkDir, float w1, float w2, float w3, float alpha, float beta, float gama, float mu);
	//static void getGrabMask(CMat edge1u, Mat &grabMask);
	//static void convexHullOfMask(CMat &mask1u, PointSeti &hullPnts);
	//static void Demo3(CMat& src, float* unary, Mat& dst, 
	//	float w1, float w2, float w3, float alpha, 
	//	float beta, float gama, float mu, int iter = 4);

	//static void runGrabCutOpenCV(CStr &wkDir);

	// Return number of difference and then expand fMask to get mask1u.
	//void ExpandMask(CStr &saveName, int expandRatio = 5);

	cv::Mat getRes();

private:
	
	void fit(float* unary);

	// Update hard segmentation after running GraphCut, 
	// Returns the number of pixels that have changed from foreground to background or vice versa.
	//int updateHardSegmentation();		

	//void initGraph();	// builds the graph for GraphCut
	
	//Mat getGrabMask(CMat &img3u, Rect rect);

	//static void getGrabMask(CMat edge1u, Mat &grabMask);
	//static void illuProb(CMat sampleDf, CmGMM &bGMM, CmGMM &fGMM, CStr &nameNE);

private:
	int _n;	// Width and height of the source image
	//Mat _imgBGR3f, _img3u; // BGR images is used to find GMMs and Lab for pixel distance , _imgLab3f
	//Mat _trimap1i;	// Trimap value
	cv::Mat _segVal1f;	// Hard segmentation with type SegmentationValue
	cv::Mat _unary2f; // Unary energies for segmentation
	
	//CmGMM _bGMM, _fGMM; // Background and foreground GMM
	//Mat _bGMMidx1i, _fGMMidx1i;	// Background and foreground GMM components, supply memory for GMM, not used for Grabcut 
	//Mat _show3u; // Image for display medial results
	//Mat _res1u;
	//vecM _mapsF, _mapsB;

	//Mat _borderMask1u; // Mask for border regions

	DenseCRF2D _crf;
	bool usePixel_;
	//CStr _nameNE;

	//SaliencyGrabCut _Sal;
};

