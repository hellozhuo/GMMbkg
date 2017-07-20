//Author: Zhuo Su, in Beihang University (BUAA)
//date: 04/2017
//C implementation of 
//L. Vincent. Morphological grayscale reconstruction in image analysis: applications and efficient algorithms.TIP, 2(2) : 176¨C201, 1993.

void imreconstruct(
	unsigned char* srca,
	unsigned char* srcb,
	int src_conn,
	int rows,
	int cols,
	unsigned char* dst
	);

