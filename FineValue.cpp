
#include"FineValue.h"

#include "GrabCutMF.h"

float * FineValue::classify(const cv::Mat& note){
	// 	const float u_energy = -log(1.0f / M);
	// 	const float n_energy = -log((1.0f - GT_PROB) / (M - 1));
	// 	const float p_energy = -log(GT_PROB);
	int W = note.cols;
	int H = note.rows;
	float * res = new float[W*H*2];
	float* pd = (float*)note.data;
	for (int k = 0; k < W*H; k++){
		// Set the energy
		float * r = res + k*2;
		float* p = pd + k;
		float pro = *p;
		float t0 = pro + 1e-5;// 2.0 * pro / (1 * pro + 1) + 1e-5;
		float t1 = 1 - pro + 1e-5;// (1.0 - pro) / (1 * pro + 1) + 1e-5;
		//float t0 = 0.1 / 2 + (*p)*0.8;
		//float t1 = 0.1 / 2 + (1 - (*p))*0.8;
		//float t3 = 0.1 / (M - 2);
		r[0] = log(t0);
		r[1] = log(t1);
		//for (int j = 2; j < M; j++)
		//{
		//	r[j] = -log(t3);
		//}
		//         if((*p)>0.99) *p = 0.99;
		//         else if(*p<0.01) *p = 0.01;
		// 		r[0] = -log((*p) );
		// 		r[1] = -log((1 - (*p)));
	}

	return res;
}

void FineValue::getFineVal(InitValue& initVal, cv::Mat& unaryMap)
{
	float* unary = classify(unaryMap);
	proceed(initVal, unary,_iter);
	if (unary) delete[] unary;
	return;
}

void FineValue::proceed(InitValue& initVal, float* unary, int iter)
{
	// Setup the CRF model
	//DenseCRF2D crf(W, H, M);

	//GrabCutMF cutMF(initVal.m_info.numlabels, initVal.m_info.features,
	//	_w1, _w2, _w3, _alpha, _beta, _gama, _mu);

	//cutMF.initialize(unary);
	//cutMF.refine(iter);
	//resLabels = cutMF.getRes();
}