#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <cmath>

const double PI = 3.14159265358979323846;
const long STEP_NUM = 32768 * 32768;
const double STEP_LENGTH = 1.0 / STEP_NUM;
const int THREAD_NUM = 512;
const int BLOCK_NUM = 64;
const int NREPEAT = 50;

__global__ void integrate(float *globalSum, int stepNum, float stepLength, int threadNum, int blockNum)
{
    int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;
    int start = (stepNum / (blockNum * threadNum)) * globalThreadId;
    int end = (stepNum / (blockNum * threadNum)) * (globalThreadId + 1);
    int localThreadId = threadIdx.x;
    int blockId = blockIdx.x;

    __shared__ float blockSum[THREAD_NUM];

    memset(blockSum, 0, threadNum * sizeof(float));

    double x;
    for(int i = start; i < end; i++)
    {
        x = (i + 0.5) * stepLength;
        blockSum[localThreadId] += 1.0 / (1.0 + x * x);
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

__global__ void sumReduce(double *sum, double *sumArray, int arraySize)
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

    std::cout << "Configuring device..." << std::endl;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if(error != cudaSuccess)
    {
        std::cout << "cudaGetDeviceCount returned" << (int)error << std::endl;
        std::cout << cudaGetErrorString(error) << std::endl;
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return 1;
    }

    if(deviceCount == 0)
    {
        std::cout << "There are no available CUDA devices(s)" << std::endl;
        return 1;
    }
    else
    {
        std::cout << "CUDA Capable device(s) detected " << deviceCount << std::endl;
    }

    double pi = 0.0;
    double *deviceBlockSum;
    double *devicePi;

    //allocate memory on GPU (device)
    cudaMalloc((void **) &devicePi, sizeof(double));
    cudaMalloc((void **) &deviceBlockSum, sizeof(double) * BLOCK_NUM);

    //timer 
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, 0);
    std::cout << "Approximate pi using a Riemann sum" << std::endl;
    std::cout << std::endl;

    std::cout << "Running CUDA pi approximation" << std::endl;

    for(int repeat = 0; repeat < NREPEAT; repeat++)
    {
        integrate<<<BLOCK_NUM, THREAD_NUM>>>(deviceBlockSum, STEP_NUM, STEP_LENGTH, THREAD_NUM, BLOCK_NUM);
        sumReduce<<<1, BLOCK_NUM>>>(devicePi, deviceBlockSum, BLOCK_NUM);

        if(repeat == (NREPEAT - 1))
        {
            //get result to host from device
            cudaMemcpy(&pi, devicePi, sizeof(double), cudaMemcpyDeviceToHost);

            std::cout << "\tpi = " << std::setprecision(20) << pi << std::endl;
            std::cout << "\terror = " << std::fixed << fabs(pi - PI) << std::endl;
        }
    }
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, startTime, stopTime);

	std::cout << "Time elapsed to get the result: " << gpuTime / 1000 << " seconds" << std::endl;
	std::cout << std::endl;

    //free memory
    cudaFree(deviceBlockSum);

    cudaDeviceReset();
    return 0;
}