#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

const float PI = 3.1415926535897932;
const long STEP_NUM = 32768 * 32768;
const float STEP_LENGTH = 1.0 / STEP_NUM;
const int THREAD_NUM = 512;
const int BLOCK_NUM = 64;

__global__ void integrate(float *globalSum, int stepNum, float stepLength, int threadNum, int blockNum)
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

int main()
{
    int deviceCount = 0;

    printf("\nConfiguring device...\n");

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if(error != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return 1;
    }

    if(deviceCount == 0)
    {
        printf("There are no available CUDA device(s)\n");
        return 1;
    }
    else
    {
        printf("%d CUDA Capable device(s) detected\n", deviceCount);
    }

    float pi = 0.0;
    float *deviceBlockSum;
    float *devicePi;

    //allocate memory on GPU (device)
    cudaMalloc((void **) &devicePi, sizeof(float));
    cudaMalloc((void **) &deviceBlockSum, sizeof(float) * BLOCK_NUM);

    //timer 
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, 0);
    printf("Approximate pi using a Riemann sum...\n");
    integrate<<<BLOCK_NUM, THREAD_NUM>>>(deviceBlockSum, STEP_NUM, STEP_LENGTH, THREAD_NUM, BLOCK_NUM);
    sumReduce<<<1, BLOCK_NUM>>>(devicePi, deviceBlockSum, BLOCK_NUM);

    //get result to host from device
    cudaMemcpy(&pi, devicePi, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startTime, stopTime);

    printf("Running CUDA pi approximation...\n");
    printf("PI = %.16lf with error %.16lf\nTime elapsed : %f seconds.\n\n", pi, fabs(pi - PI), gpuTime / 1000);
    assert(fabs(pi - PI) <= 0.001);

    //free memory
    cudaFree(deviceBlockSum);

    cudaDeviceReset();
    return 0;
}