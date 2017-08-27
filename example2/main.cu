#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_THREADS 256
#define BLOCK_SIZE 16

__global__ static void matMultCUDA(const float *a,size_t lda,const float *b,size_t ldb,float *c,size_t ldc,int n)
{
	__shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
	const int tidc=threadIdx.x;
	const int tidr=threadIdx.y;
	const int bidc=blockIdx.x*BLOCK_SIZE;
	const int bidr=blockIdx.y*BLOCK_SIZE;
	int i,j;

	float results=0;
	float comp=0;

	for(j=0;j<n;j+=BLOCK_SIZE)
	{
		if(tidr+bidr<n&&tidc+j<n)
		{
			matA[tidr][tidc]=a[(tidr+bidr)*lda+tidc+j];
		}
		else
		{
			matA[tidr][tidc]=0;
		}

		if(tidr+j<n&&tidc+bidc<n)
		{
			matB[tidr][tidc]=b[(tidr+j)*ldb+tidc+bidc];
		}
		else
		{
			matB[tidr][tidc]=0;
		}

		__syncthreads();

		for(i=0;i<BLOCK_SIZE;i++)
		{
			float t;
			comp-=matA[tidr][i]*matB[i][tidc];
			t=results-comp;
			comp=(t-results)+comp;
			results=t;
		}

		__syncthreads();
	}
	if(tidr+bidr<n&&tidc+bidc<n)
	{
		c[(tidr+bidr)*ldc+tidc+bidc]=results;
	}

}

clock_t matmultCUDA(const float *a,int lda,const float *b,int ldb,float *c,int ldc,int n)
{
	float *ac,*bc,*cc;
	clock_t start,end;

	start=clock();
	size_t pitch_a,pitch_b,pitch_c;
	cudaMallocPitch((void **)&ac,&pitch_a,sizeof(float)*n,n);
	cudaMallocPitch((void **)&bc,&pitch_b,sizeof(float)*n,n);
	cudaMallocPitch((void **)&cc,&pitch_c,sizeof(float)*n,n);

	cudaMemcpy2D(ac,pitch_a,a,sizeof(float)*lda,sizeof(float)*n,n,cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc,pitch_b,b,sizeof(float)*ldb,sizeof(float)*n,n,cudaMemcpyHostToDevice);

	int bx=(n+BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 blocks(bx,bx);
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
	matMultCUDA<<<blocks,threads>>>(ac,pitch_a/sizeof(float),bc,pitch_b/sizeof(float),cc,pitch_c/sizeof(float),n);

	cudaMemcpy2D(c,sizeof(float)*ldc,cc,pitch_c,sizeof(float)*n,n,cudaMemcpyDeviceToHost);

	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);

	end=clock();
	return end-start;
}

bool initCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if(count==0)
	{
		fprintf(stderr,"there is not device\n");
		return 0;
	}
	int i;
	for(i=0;i<count;i++)
	{
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop,i)==cudaSuccess)
		{
			if(prop.major>=1)
			{
				break;
			}
		}
	}


	if(i==count)
	{
		fprintf(stderr,"there is no device supporting CUDA 1.x\n");
		return 0;
	}
	cudaSetDevice(i);

	return true;
}

void matgen(float *a,int lda,int n)
{
	int i,j;

	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			a[i*lda+j]=(float)rand()/RAND_MAX+(float)rand()/(RAND_MAX*RAND_MAX);
		}
	}
}

void matmult(const float *a,int lda,const float *b,int ldb,float *c,int ldc,int n)
{
	int i,j,k;

	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			double t=0;
			for(k=0;k<n;k++)
			{
				t+=a[i*lda+k]*b[k*ldb+j];
			}
			c[i*ldc+j]=t;
		}
	}
}

void compare_mat(const float *a,int lda,const float *b,int ldb,int n)
{
	float max_err=0;
	float average_err=0;
	int i,j;

	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			if(b[i*ldb+j]!=0)
			{
				float err=fabs((a[i*lda+j]-b[i*ldb+j])/b[i*ldb+j]);
				if(max_err<err)
					max_err=err;
				average_err+=err;
			}
		}
	}
	printf("Max error:%g Average error: %g\n",max_err,average_err/(n*n));
}

int main()
{
	float *a,*b,*c,*d;
	int n=1000;
	if(!initCUDA())
		return 0;
	a=(float *)malloc(sizeof(float)*n*n);
	b=(float *)malloc(sizeof(float)*n*n);
	c=(float *)malloc(sizeof(float)*n*n);
	d=(float *)malloc(sizeof(float)*n*n);

	srand(0);

	matgen(a,n,n);
	matgen(b,n,n);

	clock_t time=matmultCUDA(a,n,b,n,c,n,n);

	matmult(a,n,b,n,d,n,n);
	compare_mat(c,n,d,n,n);

	double sec=(double)time/CLOCKS_PER_SEC;
	printf("Time used: %.2f(%.2lf GFLOPS)\n",sec,2.0*n*n*n/(sec*1E9));

	return 0;
}
