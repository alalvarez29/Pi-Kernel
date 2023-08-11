#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 32768 * 32768

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