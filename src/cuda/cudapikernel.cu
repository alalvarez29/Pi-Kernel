#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>

const long STEP_NUM = 32768 * 32768;
const float STEP_LENGTH = 1.0 / STEP_NUM;
const int THREAD_NUM = 512;
const int BLOCK_NUM = 64;

/*
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
*/

__global__ void integrate(float *globalSum, int stepNum, float, stepLength, int threadNum, int blockNum)
{
    int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;
    int start = (stepNum / (blockNum * threadNum)) * globalThreadId;
    int end = (stepNum / (blockNum * threadNum)) * (globalThreadId + 1);
    int localThreadId = threadIdx.x;
    int blockId = blockIdx.x;

    __shared__ float blockSum[THREAD_NUM];

    memset(blockSum, 0, threadNum * sizeof(float));

    float x;
    for(int i = start; i < end; i++)
    {
        x = (i + 0.5f) * stepLength;
        blockSum[localThreadId] += 1.0f / (1.0f + x * x);
    }
    blockSum[localThreadId] *= stepLength * 4;

    __syncthreads();

    for(int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if(localThreadId < i)
        {
            blockSum[localThreadId] += blockSum[localThreadId + i];
        }

        __syncthreads();
    }

    if(localThreadId == 0)
    {
        globalSum[blockId] = blockSum[0];
    }
}

__global__ void sumReduce(float *sum, float *sumArray, int arraySize)
{
    int localThreadId = threadIdx.x;

    for(int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (localThreadId < i)
        {
            sumArray[localThreadId] += sumArray[localThreadId + i];
        }

        __syncthreads();
    }

    if(localThreadId == 0)
    {
        *sum = sumArray[0];
    }
}

