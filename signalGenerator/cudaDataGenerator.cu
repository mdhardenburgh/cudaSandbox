#include "dataGenerator.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void generateSineCuda(float* y_0, float phase, float frequency, float amplitude, uint32_t numSamples)
{
    uint32_t currentThreadId = threadIdx.x + (blockDim.x * blockIdx.x);
    uint32_t numThreadsInGrid = blockDim.x * gridDim.x;

    for(uint32_t n = currentThreadId; n < numSamples; n += numThreadsInGrid)
    {
        y_0[n] = (amplitude * cosf((frequency*n) + phase));         
    }
}

void DataGenerator::launchSineKernel(float* y_0, float phase, float frequency, float amplitude, uint32_t numBlocks, uint32_t blockSize)
{
    generateSineCuda<<<numBlocks, blockSize>>>(y_0, phase, frequency, amplitude, numSamples);
    cudaDeviceSynchronize();
    return;
}