#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1048576
#define BLOCK_NUM 32
#define THREAD_NUM 256
int data[DATA_SIZE];

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

void generateNumbers(int *number,int size)
{
	for(int i=0;i<size;i++)
	{
		number[i]=rand()%10;
	}
}

__global__ static void sumOfSquares(int *num,int *result,clock_t* time)
{
	extern __shared__ int shared[];
	const int tid=threadIdx.x;
	const int bid=blockIdx.x;
	int i;
	int offset=1,mask=1;
	if(tid==0)
		time[bid]=clock();
	shared[tid]=0;
	for(i=bid*THREAD_NUM+tid;i<DATA_SIZE;i+=BLOCK_NUM*THREAD_NUM)
	{
		shared[tid]+=num[i]*num[i];
	}

	__syncthreads();

	if(tid < 128)
	{
		shared[tid] += shared[tid + 128];
	}
	__syncthreads();
	if(tid < 64)
	{
		shared[tid] += shared[tid + 64];
	}
	__syncthreads();
	if(tid < 32)
	{
		shared[tid] += shared[tid + 32];
	}
	__syncthreads();
	if(tid < 16)
	{
		shared[tid] += shared[tid + 16];
	}
	__syncthreads();
	if(tid < 8)
	{
		shared[tid] += shared[tid + 8];
	}
	__syncthreads();
	if(tid < 4)
	{
		shared[tid] += shared[tid + 4];
	}
	__syncthreads();
	if(tid < 2)
	{
		shared[tid] += shared[tid + 2];
	}
	__syncthreads();
	if(tid < 1)
	{
		shared[tid] += shared[tid + 1];
	}
	__syncthreads();

	if(tid==0)
	{
		result[bid]=shared[0];
		time[bid+BLOCK_NUM]=clock();
	}
}

int main()
{
	if(!initCUDA())
	{
		return 0;
	}
	printf("CUDA initialized.\n");

	generateNumbers(data,DATA_SIZE);

	int *gpudata,*result;
	clock_t *time;
	cudaMalloc((void **)&gpudata,sizeof(int)*DATA_SIZE);
	cudaMalloc((void **)&result,sizeof(int)*BLOCK_NUM);
	cudaMalloc((void **)&time,sizeof(clock_t)*BLOCK_NUM*2);
	cudaMemcpy(gpudata,data,sizeof(int)*DATA_SIZE,cudaMemcpyHostToDevice);

	sumOfSquares<<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(int)>>>(gpudata,result,time);

	int sum[BLOCK_NUM];
	clock_t time_used[BLOCK_NUM*2];
	cudaMemcpy(&sum,result,sizeof(int)*BLOCK_NUM,cudaMemcpyDeviceToHost);
	cudaMemcpy(&time_used,time,sizeof(clock_t)*BLOCK_NUM*2,cudaMemcpyDeviceToHost);
	cudaFree(gpudata);
	cudaFree(result);
	cudaFree(time);

	int final_sum=0;
	for(int i=0;i<BLOCK_NUM;i++)
	{
		final_sum+=sum[i];
	}

	clock_t min_start,max_end;
	min_start=time_used[0];
	max_end=time_used[BLOCK_NUM];
	for(int i=1;i<BLOCK_NUM;i++)
	{
		if(min_start>time_used[i])
			min_start=time_used[i];
		if(max_end<time_used[i+BLOCK_NUM])
			max_end=time_used[i+BLOCK_NUM];
	}

	printf("sum: %d time: %d\n",final_sum,max_end-min_start);

	final_sum=0;
	for(int i=0;i<DATA_SIZE;i++)
	{
		final_sum+=data[i]*data[i];
	}
	printf("sum (CPU):%d\n",final_sum);

	return 0;
}
