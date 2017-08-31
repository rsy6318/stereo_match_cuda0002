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
#include <helper_functions.h>
#include <GL/freeglut.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glx.h>

#define uint unsigned int
#define uchar unsigned char
using namespace std;
using namespace cv;
#define block_size_x 2
#define block_size_y 64

#define rows 992   //948*1500
#define cols 1420
#define disp_max 160

#define step 1

uchar (*temp)[cols];

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

/*
__global__ void stereo_kernel(uint (*a)[cols],uint (*b)[cols],uchar (*disp)[cols])
{
	const uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	if(y<disp_max)
	{
		disp[x][y]=0;
	}
	else
	{
		disp[x][y]=0;
		uint cost=abs((float)(a[x][y]-b[x][y]));
		uint cost_now;
		for(int d=1;d<disp_max+1;d+=step)
		{
			cost_now=abs((float)(a[x][y]-b[x][y-d]));
			if(cost>cost_now)
			{
				disp[x][y]=d;
				cost=cost_now;
			}
		}
	}
}*/


__global__ void stereo_kernel(uint (*a)[cols],uint (*b)[cols],uchar (*disp)[cols])
{
	const uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	__shared__ uint sh_left[block_size_x][block_size_y];
	__shared__ uint sh_right[block_size_x][block_size_y+disp_max];

	int y1=blockIdx.y*blockDim.y+threadIdx.y-disp_max;
	int y2=blockIdx.y*blockDim.y+threadIdx.y+64-disp_max;
	int y3=blockIdx.y*blockDim.y+threadIdx.y+128-disp_max;

	if(y<disp_max)
	{
		disp[x][y]=0;
	}
	else
	{
		sh_left[threadIdx.x][threadIdx.y]=a[x][y];

		if(threadIdx.y<32)
		{
			sh_right[threadIdx.x][threadIdx.y]         =b[x][y1];
			sh_right[threadIdx.x][threadIdx.y+64]      =b[x][y2];
			sh_right[threadIdx.x][threadIdx.y+128]     =b[x][y3];
			sh_right[threadIdx.x][threadIdx.y+disp_max]=b[x][y];
		}
		else
		{
			sh_right[threadIdx.x][threadIdx.y]         =b[x][y1];
			sh_right[threadIdx.x][threadIdx.y+64]      =b[x][y2];
			sh_right[threadIdx.x][threadIdx.y+disp_max]=b[x][y];
		}

		__syncthreads();
		disp[x][y]=0;
		uint cost=abs((float)(sh_left[threadIdx.x][threadIdx.y])-(sh_right[threadIdx.x][threadIdx.y+disp_max])); //(float)(b[x][y])
		uint cost_now;
		for(int d=1;d<disp_max+1;d+=step)
		{
			cost_now=abs((float)(sh_left[threadIdx.x][threadIdx.y])-(sh_right[threadIdx.x][threadIdx.y+(disp_max-d)]));  //(float)(b[x][y-d])
			if(cost>cost_now)
			{
				disp[x][y]=d;
				cost=cost_now;
			}
		}
	}
}

__global__ void box_x(uchar (*input)[cols],uchar (*output)[cols],int win_radius)
{
	const uint idx= (blockIdx.x*blockDim.x) + threadIdx.x;
	const uint idy = (blockIdx.y*blockDim.y) + threadIdx.y;
	uint scale=(win_radius<<1)+1;
	if ((idx >= win_radius) && (idx < rows - 1 - win_radius) && (idy >= win_radius) && (idy < cols - 1 - win_radius))
	{
		uint sum=0;
		for (int x = idx-win_radius; x <idx+win_radius+1 ; x++)
		{
			sum += input[x][idy];
		}
		output[idx][idy]=sum/scale;
	}
	else
		output[idx][idy]=input[idx][idy];
}

__global__ void box_y(uchar (*input)[cols],uchar (*output)[cols],int win_radius)
{
	const uint idx= (blockIdx.x*blockDim.x) + threadIdx.x;
	const uint idy = (blockIdx.y*blockDim.y) + threadIdx.y;
	uint scale=(win_radius<<1)+1;
	if ((idx >= win_radius) && (idx < rows - 1 - win_radius) && (idy >= win_radius) && (idy < cols - 1 - win_radius))
	{
		uint sum=0;
		for (int y = idy-win_radius; y <idy+win_radius+1 ; y++)
		{
			sum += input[idx][y];
		}
		output[idx][idy]=sum/scale;
	}
	else
		output[idx][idy]=input[idx][idy];
}

void box_filter(uchar (*input)[cols],uchar (*output)[cols],int win_radius,dim3 grid_size,dim3 block_size)
{
	box_x<<<grid_size,block_size>>>(input,temp,win_radius);
	box_y<<<grid_size,block_size>>>(temp,output,win_radius);
}

int main()
{
	//cudaSetDevice(0);
	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, 0);
	//deviceProp.unifiedAddressing=0;

	dim3 threads(block_size_x,block_size_y);
	dim3 blocks(iDivUp(rows,block_size_x),iDivUp(cols,block_size_y));

	uint (*cpu_p1)[cols];
	uint (*cpu_p2)[cols];
	uint (*gpu_p1)[cols];
	uint (*gpu_p2)[cols];
	uchar (*gpu_p3)[cols];
	uchar (*gpu_p4)[cols];

	Mat im1,im2,im3;
	im3.create(rows,cols,CV_8UC1);
	im1=imread("im0.png");
	im2=imread("im1.png");

	imshow("左图",im1);

	cudaHostAlloc( (void**)&cpu_p1,rows*cols*sizeof(uint),cudaHostAllocDefault);
	cudaHostAlloc( (void**)&cpu_p2,rows*cols*sizeof(uint),cudaHostAllocDefault);

	for(int x=0;x<rows;x++)
	{
		for(int y=0;y<cols;y++)
		{
			cpu_p1[x][y]=im1.at<Vec3b>(x,y)[0]+(im1.at<Vec3b>(x,y)[1]<<8)+(im1.at<Vec3b>(x,y)[2]<<16);
			cpu_p2[x][y]=im2.at<Vec3b>(x,y)[0]+(im2.at<Vec3b>(x,y)[1]<<8)+(im2.at<Vec3b>(x,y)[2]<<16);
		}
	}

	//自动补位
	size_t pitch;
	cudaMallocPitch(&gpu_p1,&pitch,cols*sizeof(uint),rows);
	cudaMallocPitch(&gpu_p2,&pitch,cols*sizeof(uint),rows);
	cudaMallocPitch(&gpu_p3,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch(&gpu_p4,&pitch,cols*sizeof(uchar),rows);
	cudaMallocPitch(&temp,&pitch,cols*sizeof(uchar),rows);

	cudaMemcpyAsync(gpu_p1,cpu_p1,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
	cudaMemcpyAsync(gpu_p2,cpu_p2,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);

	stereo_kernel<<<blocks,threads>>>(gpu_p1,gpu_p2,gpu_p3);

	box_filter(gpu_p3,gpu_p4,2,blocks,threads);

	cudaMemcpy(im3.data,gpu_p4,rows*cols*sizeof(uchar),cudaMemcpyDeviceToHost);

	//medianBlur(im3,im3,3);
	//blur(im3,im3,Size(3,3),Point(-1,-1));
	imshow("视差图",im3);
	imwrite("disp.bmp",im3);
	//waitKey(0);

	return 0;
}
