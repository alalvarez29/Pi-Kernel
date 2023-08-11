#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

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
            std::cout << "Error at line: " << fileName std::endl;
            std::cout << line << std::endl;
            std::cout << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
    #endif
}

__global__ void integration(int *n, float *g_sum)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    __shared__ float s_sum[THREADS];

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