//#include "stdafx.h"
#include "GrabCutMF.h"


GrabCutMF::GrabCutMF(int N, unsigned int** features, float w1, float w2, float w3, float alpha, float beta, float gama, float mu)
	:_crf(N, 2), _n(N)
{
	CV_Assert(N > 0 && features);
	_segVal1f.create(1, _n, CV_32FC1);
	_unary2f.create(1, _n, CV_32FC2);
	if (w1 != 0)
		_crf.addPairwiseBilateral(alpha, alpha, beta, beta, beta, features, w1);
	if (w2 != 0)
		_crf.addPairwiseGaussian(gama, gama, features, w2);
	if (w3 != 0)
		_crf.addPairwiseColorGaussian(mu, mu, mu, features, w3);
}

// Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper 
void GrabCutMF::initialize(float* unary)
{
	CV_Assert(unary);
	fit(unary);
}


void GrabCutMF::fit(float* unary)
{
	Vec2f* unryV = _unary2f.ptr<Vec2f>(0);
	for (int i = 0; i < _n; i++){
		float* pd = unary + i * 2;
		float prb1 = *pd;
		float prob2 = *(pd + 1);

		unryV[i] = Vec2f(prb1, prob2);
	}
}


cv::Mat GrabCutMF::getRes()
{
	return _segVal1f;
}

void GrabCutMF::refine(int iter)
{
	// Initialize _unary1f using GMM
	_crf.setUnaryEnergy(_unary2f.ptr<float>(0));
	float* prob = _crf.binarySeg(iter, 1.f);
	float* res = (float*)_segVal1f.data;
	const int N = _n;
	for(int i=0; i<N; i++, prob+=2)
		res[i] = prob[1]/(prob[0]+prob[1]+1e-20f);

	return;// _segVal1f;
}

//Mat GrabCutMF::showMedialResults(CStr& title)
//{
//	_show3u.create(_h, _w, CV_8UC3);
//	_imgBGR3f.convertTo(_show3u, CV_8U, 255);
//
//	for (int y = 0; y < _h; y++){
//		const int* triVal = _trimap1i.ptr<int>(y);
//		const float* segVal = _segVal1f.ptr<float>(y);
//		Vec3b* triD = _show3u.ptr<Vec3b>(y);
//		for (int x = 0; x < _w; x++, triD++) {
//			switch (triVal[x]){
//				case UserFore: (*triD)[2] = 255; break; // Red
//				case UserBack: (*triD)[1] = 255; break; // Green
//			}
//			if (segVal[x] < 0.5){
//				(*triD)[0] = 255;
//				if (x-1 >= 0 && segVal[x-1] > 0.5 || x+1 < _w && segVal[x+1] > 0.5)
//					(*triD) = Vec3b(0,0,255);
//				if (y-1 >= 0 && _segVal1f.at<float>(y-1, x) > 0.5 || y+1 < _h && _segVal1f.at<float>(y+1, x) > 0.5)
//					_show3u.at<Vec3b>(y,x) = Vec3b(0,0,255);				
//			}
//		}
//	}
//	CmShow::SaveShow(_show3u, title);
//	return _show3u;
//}

//void GrabCutMF::convexHullOfMask(CMat &mask1u, PointSeti &hullPnts)
//{
//	const int H = mask1u.rows - 1, W = mask1u.cols - 1;
//	PointSeti pntSet;
//	pntSet.reserve(H*W);
//	for (int r = 1; r < H; r++){
//		const byte* m = mask1u.ptr<byte>(r);
//		for (int c = 1; c < W; c++){
//			if (m[c] < 200)
//				continue;
//			if (m[c-1] < 200 || m[c + 1] < 200 || mask1u.at<byte>(r-1, c) < 200 || mask1u.at<byte>(r+1, c) < 200)
//				pntSet.push_back(Point(c, r));
//		}
//	}
//	convexHull(pntSet, hullPnts);
//}

//void GrabCutMF::getGrabMask(CMat edge1u, Mat &grabMask)
//{
//	//imshow("Edge map", edge1u);
//	//imshow("Grabmask", grabMask);
//	//waitKey(1);
//
//	queue<Point> selectedPnts;
//	int _w = edge1u.cols, _h = edge1u.rows;
//	for (int y = 1, maxY = _h - 1, stepSz = edge1u.step.p[0]; y < maxY; y++) {
//		byte* m = grabMask.ptr<byte>(y);
//		for (int x = 1, maxX = _w - 1; x < maxX; x++)
//			if (m[x] == 255 && (m[x - 1] == 0 || m[x + 1] == 0 || m[x - stepSz] == 0 || m[x + stepSz] == 0))
//				selectedPnts.push(Point(x, y));
//	}
//
//	// Flood fill
//	while (!selectedPnts.empty()){
//		Point crntPnt = selectedPnts.front();
//		grabMask.at<byte>(crntPnt) = 255;
//		selectedPnts.pop();
//		for (int i = 0; i < 4; i++){
//			Point nbrPnt = crntPnt + DIRECTION4[i];
//			if (CHK_IND(nbrPnt) && grabMask.at<byte>(nbrPnt) == 0 && edge1u.at<byte>(nbrPnt) == 0)
//				grabMask.at<byte>(nbrPnt) = 255, selectedPnts.push(nbrPnt);
//		}
//	}
//
//
//	//imshow("Grabmask New", grabMask);
//	//waitKey(0);
//}


//Mat GrabCutMF::getGrabMask(CMat &img3u, Rect rect)
//{
//	// Initialize flood fill
//	queue<Point> selectedPnts;
//	const int _h = img3u.rows, _w = img3u.cols, BW = 5;
//	{// If not connected to image border, expand selection border unless stopped by edges
//		Point rowT(rect.x, rect.y), rowB(rect.x, rect.y + rect.height - 1);
//		Point colL(rect.x, rect.y), colR(rect.x + rect.width - 1, rect.y);
//		if (rect.x >= BW) // Expand left edge
//			for (int y = 0; y < rect.height; y++, colL.y++) selectedPnts.push(colL);
//		else
//			rect.x = BW;
//		if (rect.y >= BW) // Expand top edge
//			for (int x = 0; x < rect.width; x++, rowT.x++)	selectedPnts.push(rowT);
//		else
//			rect.y = BW;
//		if (rect.x + rect.width + BW <= _w) // Expand right edge	
//			for (int y = 0; y < rect.height; y++, colR.y++) selectedPnts.push(colR);
//		else
//			rect.width = _w - rect.x - BW;
//		if (rect.y + rect.height + BW <= _h) // Expand bottom edge
//			for (int x = 0; x < rect.width; x++, rowB.x++) selectedPnts.push(rowB);
//		else
//			rect.height = _h - rect.y - BW;
//	}
//
//	Mat mask1u(img3u.size(), CV_8U);
//	memset(mask1u.data, 255, mask1u.step.p[0] * mask1u.rows);
//	mask1u(rect) = 0;
//
//	Mat edge1u;
//	CmCv::CannySimpleRGB(img3u, edge1u, 120, 1200, 5);
//	dilate(edge1u, edge1u, Mat(), Point(-1, -1), 3);
//	//rectangle(edge1u, rect, Scalar(128));
//	//imwrite(sameNameNE + "_Selection.png", edge1u);
//
//	// Flood fill
//	while (!selectedPnts.empty()){
//		Point crntPnt = selectedPnts.front();
//		mask1u.at<byte>(crntPnt) = 255;
//		selectedPnts.pop();
//		for (int i = 0; i < 4; i++){
//			Point nbrPnt = crntPnt + DIRECTION4[i];
//			if (CHK_IND(nbrPnt) && mask1u.at<byte>(nbrPnt) == 0 && edge1u.at<byte>(nbrPnt) == 0)
//				mask1u.at<byte>(nbrPnt) = 255, selectedPnts.push(nbrPnt);
//		}
//	}
//	CmCv::rubustifyBorderMask(mask1u(Rect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2)));
//	return mask1u;
//}


//Mat GrabCutMF::drawResult()
//{
//	compare(_segVal1f, 0.5, _res1u, CMP_GT); 
//	int ts = _segVal1f.type();
//	int tr = _res1u.type();
//	return _res1u;
//}

//void GrabCutMF::runGrabCutOpenCV(CStr &wkDir)
//{
//	CStr imgDir = wkDir + "Imgs/", salDir = wkDir + "Sal/";
//	vecS namesNE;
//	int imgNum = CmFile::GetNamesNE(imgDir + "*.jpg", namesNE);
//	CmFile::MkDir(salDir);
//
//	// Number of labels
//	CmTimer tm("Time");
//	tm.Start();
//	for (int i = 0; i < imgNum; i++){
//		printf("Processing %d/%d: %s.jpg%20s\r", i, imgNum, _S(namesNE[i]), "");
//		CmFile::Copy(imgDir + namesNE[i] + ".jpg", salDir + namesNE[i] + ".jpg");
//		CmFile::Copy(imgDir + namesNE[i] + ".png", salDir + namesNE[i] + "_GT.png");
//		Mat imMat3u = imread(imgDir + namesNE[i] + ".jpg");
//		Mat gt1u = imread(imgDir + namesNE[i] + ".png", CV_LOAD_IMAGE_GRAYSCALE);
//		imwrite(imgDir + namesNE[i] + ".bmp", gt1u);
//		blur(gt1u, gt1u, Size(3,3));
//		Rect wkRect = CmCv::GetMaskRange(gt1u, 1, 128);
//
//		// Prepare data for OneCut
//		//Mat rectImg = Mat::ones(gt1u.size(), CV_8U)*255;
//		//rectImg(wkRect) = 0;
//		//imwrite(salDir + namesNE[i] + ".bmp", imMat3u);
//		//imwrite(salDir + namesNE[i] + "_t.bmp", rectImg);
//
//		Mat res1u, bgModel, fgModel;
//		grabCut(imMat3u, res1u, wkRect, bgModel, fgModel, 1, GC_INIT_WITH_RECT);
//		grabCut(imMat3u, res1u, wkRect, bgModel, fgModel, 5);
//		compare(res1u, GC_PR_FGD, res1u, CMP_EQ);
//		imwrite(salDir + namesNE[i] + "_GC.png", res1u);
//	}
//	tm.Stop();
//	double avgTime = tm.TimeInSeconds()/imgNum;
//	printf("Speed: %gs, %gfps\t\t\n", avgTime, 1/avgTime);
//
//	CmEvaluation::EvalueMask(imgDir + "*.png", salDir, "GC", "");
//}

//void GrabCutMF::Demo3(CMat& src, float* unary, Mat& dst, float w1, float w2, float w3, float alpha, float beta, float gama, float mu, int iter)
//{
	//assert(src.type() == CV_8UC3);

	//double maxWeight = 2; // 2: 0.958119, 1: 0.953818, 

	//Mat imMat3u = src, imMat3f, gt1u;

	////Mat res1u = Mat::zeros(imMat3u.size(), CV_8UC1);

	//imMat3u.convertTo(imMat3f, CV_32FC3, 1 / 255.0);

	////cv::Mat borderMask1u = cv::Mat::zeros(mask.size(),mask.type());

	//Rect rect(63, 15, 326, 237);
	//cv::Mat mask;
	////borderMask1u(rect).setTo(255);
	//GrabCutMF cutMF(imMat3f, imMat3u, "", w1, w2, w3, alpha, beta, gama, mu);

	//cutMF.initialize(rect, mask, 1, false, unary);
	//cutMF.refine(iter);

	//Mat res = cutMF.drawResult(), res1u;

	////imshow("imMat3u", imMat3u);
	////imshow("mask", mask);
	////imshow("res", res);
	////waitKey(0);
	////cv::normalize(res, dst, 0.0, 1.0, cv::NORM_MINMAX);
	//res.convertTo(dst, CV_32FC1, 1 / 255.0);
	//return;
//}



//Mat GrabCutMF::setGrabReg(const Rect &rect, CMat &bordMask1u)
//{
//	CmGrabSal sGC(_imgBGR3f, bordMask1u, _nameNE);
//	sGC.HistgramGMMs();
//	_segVal1f = sGC.GetSaliencyCues();
//
//	_trimap1i = UserBack;
//	_trimap1i(rect) = TrimapUnknown;
//
//#pragma omp parallel for
//	for (int y = 0; y < _h; y++){
//		float* segV = _segVal1f.ptr<float>(y);
//		Vec2f* unryV = _unary2f.ptr<Vec2f>(y);
//		int* triV = _trimap1i.ptr<int>(y); 
//		Vec3f* img = _imgBGR3f.ptr<Vec3f>(y);
//		for (int x = 0; x < _w; x++){
//			float prb; // User Back
//			switch (triV[x]){
//			case UserBack: prb = 0; break;
//				//case UserFore: prb = 0.9f; break;
//			default: prb = 0.8f*segV[x];
//			}
//			unryV[x] = Vec2f(prb, 1-prb);
//			segV[x] = prb;
//		}
//	}
//	if (_nameNE.size())
//		imwrite(_nameNE + "_P.png", _segVal1f*255);
//
//	return _segVal1f;
//}
