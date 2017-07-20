
#include"vincent11.h"

void imreconstruct(
	unsigned char* srca,
	unsigned char* srcb,
	int src_conn,
	int rows,
	int cols,
	unsigned char* dst
	)
{
	int n,m;
	int ii,jj,conn;
	int *dims;
	int ndims;
	const int  *vettore;
	unsigned char *II,*JJ;
	int mp1,np1,np2,mp2;
	int cordx1,cordx0;
	int cordx,cordxm1,cordxmm,cordxmmm1,cordxpmm1;
	int cordxp1,cordxpm,cordxpmp1,cordxmmp1;
	int *coda;
	int lettura;
	int scrittura;
	int max,v1,v2,v3,v4,v5,totaln;
	int coda_dim,percentuale;

	//a = srca;
	//b = srcb;
	//a=mxGetData(prhs[0]);//column oriented
	//b=mxGetData(prhs[1]);
	conn=src_conn;

	ndims=2;
	m = rows;
	n = cols;
	//vettore = mxGetDimensions(prhs[0]);
	//n=vettore[0];//rows
	//m=vettore[1];//cols
	//dims=mxCalloc(2,sizeof(int));
	dims = (int*)malloc(2 * sizeof(int));
	*(dims) = n + 2;
	*(dims + 1) = m + 2;
	II = (unsigned char *)malloc((n + 2)*(m + 2)*sizeof(unsigned char));
	JJ = (unsigned char *)malloc((n + 2)*(m + 2)*sizeof(unsigned char));
	memset(II, 0, (n + 2)*(m + 2)*sizeof(unsigned char));
	memset(JJ, 0, (n + 2)*(m + 2)*sizeof(unsigned char));
	//II= mxCalloc((n+2)*(m+2),sizeof(unsigned char));
	//JJ= mxCalloc((n+2)*(m+2),sizeof(unsigned char));

	//percentuale=30;
	coda_dim = n*m;
	//coda_dim=(n*m*percentuale)%100;
	//coda_dim=(n*m*percentuale-coda_dim)/100;
	//coda = new unsigned long[coda_dim];
	coda = (int *)malloc(coda_dim*sizeof(int));
	memset(coda, 0, coda_dim*sizeof(int));


	mp1=m+1;
	mp2=m+2;
	np1=n+1;
	np2=n+2;

	totaln = np2*mp2;

	for (jj=1;jj<mp1;jj++)
		{   for (ii=1;ii<np1;ii++)
			{cordx1=jj*np2+ii;
			 cordx0=(jj-1)*n+ii-1;
			 *(II+cordx1)=*(srca+cordx0);
			 *(JJ+cordx1)=*(srcb+cordx0);
			}
		}

	if(conn==8)
	{

	for (ii=2;ii<np2;ii++)
		{   for (jj=2;jj<mp2;jj++)
			{cordx=(jj-1)*np2+ii-1;
			 max=*(JJ+cordx);
         
			 v1=*(JJ+cordx-1);
			 if(max<v1)
			   max=v1;

			 v1=*(JJ+cordx-np2);
			 if(max<v1)
			   max=v1;

			 v1=*(JJ+cordx-np2-1); 
			 if(max<v1)
			   max=v1;

			 v1=*(JJ+cordx+np2-1);
			 if(max<v1)
			   max=v1;

			 v1=*(II+cordx);
			 if(max>v1)
			   max=v1;

			*(JJ+cordx)=max;         
   
			}
		}

	lettura=0;
	scrittura=0;

	for (ii=np1;ii>1;ii--)
		{   for (jj=mp1;jj>1;jj--)
			{cordx=(jj-1)*np2+ii-1;

			 cordxp1=cordx+1;
			 cordxpm=cordx+np2;
			 cordxpmp1=cordxpm+1;
			 cordxmmp1=cordx-np2+1;

			 max=*(JJ+cordx);
			 v1=*(JJ+cordxp1);
			 v2=*(JJ+cordxpm);
			 v3=*(JJ+cordxmmp1);
			 v4=*(JJ+cordxpmp1);
			 v5=*(II+cordx);

			 if(max<v1)
			   max=v1;

			 if(max<v2)
			   max=v2;
      
			 if(max<v3)
			  max=v3;

			 if(max<v4)
			   max=v4;

			 if(max>v5)
			   max=v5; 

        

			*(JJ+cordx)=max;

       

         

			if ((v2<max)&&(v2<*(II+cordxpm)))
			{*(coda+scrittura)=cordx;
			  scrittura=scrittura+1;
			}
			else
				{if ((v1<max)&&(v1<*(II+cordxp1)))
				 {*(coda+scrittura)=cordx;
				  scrittura=scrittura+1;
				 }
				 else
					 {if ((v4<max)&&(v4<*(II+cordxpmp1)))
					  {*(coda+scrittura)=cordx;
					   scrittura=scrittura+1;
					  }
					  else
						  {if ((v3<max)&&(v3<*(II+cordxmmp1)))
						   {*(coda+scrittura)=cordx;
							scrittura=scrittura+1;
						   }
						  }

					 }
				}


       
   
			}
		}


	while(lettura<scrittura){

	cordx=*(coda+lettura);
	lettura++;

	cordxm1=cordx-1;
	cordxmm=cordx-np2;
	cordxmmm1=cordx-np2-1;
	cordxpmm1=cordx+np2-1;
	cordxp1=cordx+1;
	cordxpm=cordx+np2;
	cordxpmp1=cordxpm+1;
	cordxmmp1=cordx-np2+1;

	max=*(JJ+cordx);

	v1=*(JJ+cordxmmm1);
	if((v1<max)&&(*(II+cordxmmm1)!=v1))
	{  if(max<*(II+cordxmmm1))
		  *(JJ+cordxmmm1)=max;
	   else
		  *(JJ+cordxmmm1)=*(II+cordxmmm1);
	   *(coda+scrittura)=cordxmmm1;
	   scrittura++;
	}

	v1=*(JJ+cordxm1);
	if((v1<max)&&(*(II+cordxm1)!=v1))
	{  if(max<*(II+cordxm1))
		  *(JJ+cordxm1)=max;
	   else
		  *(JJ+cordxm1)=*(II+cordxm1);
	   *(coda+scrittura)=cordxm1;
	   scrittura++;
	}

	v1=*(JJ+cordxpmm1);
	if((v1<max)&&(*(II+cordxpmm1)!=v1))
	{  if(max<*(II+cordxpmm1))
		  *(JJ+cordxpmm1)=max;
	   else
		  *(JJ+cordxpmm1)=*(II+cordxpmm1);
	   *(coda+scrittura)=cordxpmm1;
	   scrittura++;
	}

	v1=*(JJ+cordxmm);
	if((v1<max)&&(*(II+cordxmm)!=v1))
	{  if(max<*(II+cordxmm))
		  *(JJ+cordxmm)=max;
	   else
		  *(JJ+cordxmm)=*(II+cordxmm);
	   *(coda+scrittura)=cordxmm;
	   scrittura++;
	}

	v1=*(JJ+cordxpm);
	if((v1<max)&&(*(II+cordxpm)!=v1))
	{  if(max<*(II+cordxpm))
		  *(JJ+cordxpm)=max;
	   else
		  *(JJ+cordxpm)=*(II+cordxpm);
	   *(coda+scrittura)=cordxpm;
	   scrittura++;
	}

	v1=*(JJ+cordxmmp1);
	if((v1<max)&&(*(II+cordxmmp1)!=v1))
	{  if(max<*(II+cordxmmp1))
		  *(JJ+cordxmmp1)=max;
	   else
		  *(JJ+cordxmmp1)=*(II+cordxmmp1);
	   *(coda+scrittura)=cordxmmp1;
	   scrittura++;
	}


	v1=*(JJ+cordxp1);
	if((v1<max)&&(*(II+cordxp1)!=v1))
	{  if(max<*(II+cordxp1))
		  *(JJ+cordxp1)=max;
	   else
		  *(JJ+cordxp1)=*(II+cordxp1);
	   *(coda+scrittura)=cordxp1;
	   scrittura++;
	}

	v1=*(JJ+cordxpmp1);
	if((v1<max)&&(*(II+cordxpmp1)!=v1))
	{  if(max<*(II+cordxpmp1))
		  *(JJ+cordxpmp1)=max;
	   else
		  *(JJ+cordxpmp1)=*(II+cordxpmp1);
	   *(coda+scrittura)=cordxpmp1;
	   scrittura++;
	}

	}





	}


	if(conn==4)
	{
	for (ii=2;ii<np2;ii++)
		{   for (jj=2;jj<mp2;jj++)
			{cordx=(jj-1)*np2+ii-1;
			 max=*(JJ+cordx);
         
			 v1=*(JJ+cordx-1);
			 if(max<v1)
			   max=v1;

			 v1=*(JJ+cordx-np2);
			 if(max<v1)
			   max=v1;


			 v1=*(II+cordx);
			 if(max>v1)
			   max=v1;

			*(JJ+cordx)=max;         
   
			}
		}

	lettura=0;
	scrittura=0;

	for (ii=np1;ii>1;ii--)
		{   for (jj=mp1;jj>1;jj--)
			{cordx=(jj-1)*np2+ii-1;

			 cordxp1=cordx+1;
			 cordxpm=cordx+np2;
        

			 max=*(JJ+cordx);
			 v1=*(JJ+cordxp1);
			 v2=*(JJ+cordxpm);
      
			 v5=*(II+cordx);

			 if(max<v1)
			   max=v1;

			 if(max<v2)
			   max=v2;
      
        

			 if(max>v5)
			   max=v5; 

        

			*(JJ+cordx)=max;

       

         

			if ((v2<max)&&(v2<*(II+cordxpm)))
			{*(coda+scrittura)=cordx;
			  scrittura=scrittura+1;
			}
			else
				{if ((v1<max)&&(v1<*(II+cordxp1)))
				 {*(coda+scrittura)=cordx;
				  scrittura=scrittura+1;
				 }
             
                 
				}


       
   
			}
		}
	
	while(lettura<scrittura){

	cordx=*(coda+lettura);
	lettura++;

	cordxm1=cordx-1;
	cordxmm=cordx-np2;

	cordxp1=cordx+1;
	cordxpm=cordx+np2;


	max=*(JJ+cordx);

	v1=*(JJ+cordxm1);
	if((v1<max)&&(*(II+cordxm1)!=v1))
	{  if(max<*(II+cordxm1))
		  *(JJ+cordxm1)=max;
	   else
		  *(JJ+cordxm1)=*(II+cordxm1);

	if (scrittura < coda_dim - 1 && cordxm1%np2 > 1)
		{
			*(coda + scrittura) = cordxm1;
			scrittura++;
		}
	   
	}


	v1=*(JJ+cordxmm);
	if((v1<max)&&(*(II+cordxmm)!=v1))
	{  if(max<*(II+cordxmm))
		  *(JJ+cordxmm)=max;
	   else
		  *(JJ+cordxmm)=*(II+cordxmm);
	if (scrittura < coda_dim - 1 && cordxmm > 2 * np2)
	{
		*(coda + scrittura) = cordxmm;
		scrittura++;
	}
	   
	}


	v1=*(JJ+cordxpm);
	if((v1<max)&&(*(II+cordxpm)!=v1))
	{  if(max<*(II+cordxpm))
		  *(JJ+cordxpm)=max;
	   else
		  *(JJ+cordxpm)=*(II+cordxpm);

	if (scrittura < coda_dim-1 && cordxpm < totaln - 2*np2)
	{
		*(coda + scrittura) = cordxpm;
		scrittura++;
	}
	   
	}

	v1=*(JJ+cordxp1);
	if((v1<max)&&(*(II+cordxp1)!=v1))
	{  if(max<*(II+cordxp1))
		  *(JJ+cordxp1)=max;
	   else
		  *(JJ+cordxp1)=*(II+cordxp1);

	if (scrittura < coda_dim - 1 && cordxp1%np2 < np2 - 2)
	{
		*(coda + scrittura) = cordxp1;
		scrittura++;
	}
	   
	}



	}

	}

	for (jj=1;jj<mp1;jj++)
		{   for (ii=1;ii<np1;ii++)
			{cordx1=jj*np2+ii;
			 cordx0=(jj-1)*n+ii-1;
			 *(dst+cordx0)=*(JJ+cordx1);
			}
		}

	free(dims);
	free(II);
	free(JJ);
	free(coda);

	return; 
}

	 
