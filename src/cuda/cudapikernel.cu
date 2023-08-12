#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>

#define INTERVALS 32768 * 32768

#define THREADS 256
#define BLOCKS 64

inline void checkCUDAError(const char *fileName, const int line)
{
    #ifdef DEBUG
        cudaThreadSynchronize();
        cudaError_t = cudaGetLastError();
        if(error != cudaSuccess)
        {
            printf("Error at %s: line %i: %s\n", fileName, line, cudaGetErrorString(error));
			exit(-1); 
        }
    #endif
}

__global__ void integration(int *n, double *g_sum)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    __shared__ double s_sum[THREADS];

    double sum = 0.0;
    double step = 1.0 / (double)*n;

    for (int i = idx + 1; i <= *n; i += blockDim.x * BLOCKS)
    {
        double x = step * ((double)i - 0.5);
        sum += 4.0 / (1.0 + x*x);
    }

    s_sum[tx] = sum * step;

    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (tx < i)
        {
            s_sum[tx] += s_sum[tx + i];
        }

        __syncthreads();
    }

    g_sum[idx] = s_sum[tx];
}

__global__ static void sumReduce(int *n, double *g_sum)
{
    int tx = threadIdx.x;
    __shared__ double s_sum[THREADS];

    if (tx < BLOCKS)
    {
        s_sum[tx] = g_sum[tx * THREADS];
    }
    else
    {
        s_sum[tx] = 0.0;
    }

    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (tx < i)
        {
            s_sum[tx] += s_sum[tx + i];
        }

        __syncthreads();
    }

    g_sum[tx] = s_sum[tx];
}

int main(int argc, char *argv[])
{
    int deviceCount = 0;

    std::cout << "Approximate pi using a Riemann sum..." << std::endl;
    std::cout << std::endl;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return 1;
    }

    deviceCount == 0 ? printf("There are no available CUDA device(s)\n") : printf("%d CUDA Capable device(s) detected\n", deviceCount);

    int N = INTERVALS; 
    int *n_d;
    double pi;
    double *pi_d;

    cudaMalloc((void**) &n_d, sizeof(int) * 1);
    cudaMalloc((void**) &pi_d, sizeof(double) * BLOCKS * THREADS);

    cudaMemcpy(n_d, &N, sizeof(int) * 1, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float gpuTime;

    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    cudaEventRecord(start, 0);
	integration<<<BLOCKS,THREADS>>>(n_d, pi_d);
	checkCUDAError(__FILE__, __LINE__);
	sumReduce<<< 1, THREADS >>>(n_d, pi_d);
	checkCUDAError(__FILE__, __LINE__);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	
	cudaMemcpy(&pi, pi_d, sizeof(double) * 1, cudaMemcpyDeviceToHost);

	std::cout << "Pi is: " << pi << std::endl;  

	printf("GPU implementation time: %f ms\n", gpuTime);

	cudaFree(n_d);
	cudaFree(pi_d);

	cudaDeviceReset();
	return 0;
}