#include <iostream>
#include <math.h>
#include <cuda_profiler_api.h>
#include <string>
#include <stdio.h>
#include <cstdint>

__device__ float w0 = (1*M_PI)/34;
__device__ float w1 = (1*M_PI)/35;

__global__
void calcSine(uint32_t N, uint32_t* arrayPtr)
{
    uint32_t currentThreadId = threadIdx.x + (blockDim.x * blockIdx.x);
    uint32_t numThreadsInGrid = blockDim.x * gridDim.x;

    for(uint32_t n = currentThreadId; n < N; n += numThreadsInGrid)
    {
        arrayPtr[n] = uint32_t(( (sinf(w0*n) + sinf(w1*n + (M_PI/2)) ) + 1.0) * 127);
    }
}

int main(void)
{
    uint32_t N = 10000000;
    uint32_t* ptr_y;
    cudaMallocManaged(&ptr_y, N*sizeof(uint32_t));

    uint32_t blockSize = 256;
    uint32_t numBlocks = (N + blockSize - 1)/blockSize;

    // Run kernel on 10M elements on the CPU
    calcSine<<<numBlocks, blockSize>>>(N, ptr_y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for(uint32_t i = 0; i < N; i++)
    {
        std::cout<<ptr_y[i] << std::endl;
    }

    cudaFree(ptr_y);
    
    return 0;
}
