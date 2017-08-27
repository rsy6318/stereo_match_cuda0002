#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <helper_gl.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_string.h>
#include <helper_functions.h>
#include <GL/freeglut.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glx.h>

using namespace cv;
using namespace std;

#define uchar unsigned char
#define uint unsigned int



#define rows 948
#define cols 1500

#define rad 9
#define disp_max 305

texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex2Dleft;
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex2Dright;

__device__ unsigned int __usad4(unsigned int A, unsigned int B, unsigned int C=0)
{
    unsigned int result;
#if (__CUDA_ARCH__ >= 300) // Kepler (SM 3.x) supports a 4 vector SAD SIMD
    asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r"(result):"r"(A), "r"(B), "r"(C));
#else // SM 2.0            // Fermi  (SM 2.x) supports only 1 SAD SIMD, so there are 4 instructions
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b0, %2.b0, %3;": "=r"(result):"r"(A), "r"(B), "r"(C));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b1, %2.b1, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b2, %2.b2, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b3, %2.b3, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
#endif
    return result;
}

#define RAD 3

#define STEPS 3

#define blockSize_x 32
#define blockSize_y 32


void load_img_gray(const char *filename,unsigned char **data1,unsigned int ww,unsigned int hh)
{
	Mat img=imread(filename);
	*data1=(unsigned char *)malloc(sizeof(unsigned char)*ww*hh*4);
	unsigned int size=ww*hh;
	unsigned char *ptr=*data1;
	for(int i=0;i<size;i++)
	{
		*ptr++=*img.data++;
		*ptr++=*img.data++;
		*ptr++=*img.data++;
		*ptr++=0;
	}
}

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

#define block_size_x 16
#define block_size_y 16



/*__global__ void stereo_kernel(unsigned int *img0,unsigned int *img1,unsigned int *out)
{
	unsigned int x=blockDim.x*blockIdx.x+threadIdx.x;
	unsigned int y=blockDim.y*blockIdx.y+threadIdx.y;

	__shared__ unsigned int sh_data[block_size_x+2*rad][block_size_y+2*rad];


}*/


__global__ void check_kernel(unsigned int *out)
{
	unsigned int x=blockDim.x*blockIdx.x+threadIdx.x;
	unsigned int y=blockDim.y*blockIdx.y+threadIdx.y;

	out[x*rows+y]=tex2D(tex2Dleft,x,y);
}

int main(int argc,char **argv)
{
	cudaDeviceProp deviceProp;
	deviceProp.major = 0;
	deviceProp.minor = 0;
	int dev = 0;

	// This will pick the best possible CUDA capable device
	dev = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n", deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	unsigned char *data0;
	unsigned char *data1;
	load_img("im0.png",&data0,cols,rows);
	load_img("im1.png",&data1,cols,rows);

	unsigned int h;
	unsigned int w;

	h=948;
	w=1500;


	unsigned int *d_data0,*d_data1,*d_out;
	cudaMalloc((void **)&d_data0,rows*cols*sizeof(int));
	cudaMalloc((void **)&d_data1,rows*cols*sizeof(int));
	cudaMalloc((void **)&d_out,rows*cols*sizeof(int));

	cudaChannelFormatDesc ca_desc0=cudaCreateChannelDesc<unsigned int>();
	cudaChannelFormatDesc ca_desc1=cudaCreateChannelDesc<unsigned int>();

	tex2Dleft.addressMode[0]=cudaAddressModeClamp;
	tex2Dleft.addressMode[1]=cudaAddressModeClamp;
	tex2Dleft.filterMode=cudaFilterModePoint;
	tex2Dleft.normalized=false;
	tex2Dright.addressMode[0]=cudaAddressModeClamp;
	tex2Dright.addressMode[1]=cudaAddressModeClamp;
	tex2Dright.filterMode=cudaFilterModePoint;
	tex2Dright.normalized=false;

	cudaArray *cuarray0;
	cudaArray *cuarray1;
	cudaMallocArray(&cuarray0,&ca_desc0,rows,cols);
	cudaMallocArray(&cuarray1,&ca_desc1,rows,cols);
	cudaMemcpyToArray(cuarray0,0,0,data0,w*h*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpyToArray(cuarray1,0,0,data1,w*h*sizeof(int),cudaMemcpyHostToDevice);

	checkCudaErrors(cudaBindTextureToArray(tex2Dleft, cuarray0, ca_desc0));
	checkCudaErrors(cudaBindTextureToArray(tex2Dright, cuarray1, ca_desc1));

	dim3 block_size(block_size_x,block_size_y);
	dim3 grid_size(iDivUp(rows,block_size.x),iDivUp(cols,block_size.y));

	check_kernel<<<grid_size,block_size>>>(d_out);



	Mat out;
	out.create(rows,cols,CV_8UC1);
	unsigned int *h_out;
	h_out=(unsigned int*)malloc(w*h*4);

	cudaMemcpy(h_out,d_out,sizeof(int)*w*h,cudaMemcpyDeviceToHost);

	for(int i=0;i<w*h;i++)
	{
		*out.data++=*h_out++;
	}



	imshow("111",out);
	waitKey(0);
	return 0;
}
