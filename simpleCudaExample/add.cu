/**
 * @file add.cu
 * @brief simple cuda add sandbox
 * @author Matthew Hardenburgh
 * @date 9/13/2020
 * @copyright Matthew Hardenburgh 2020. All Rights Reserved.
 *
 * @section license LICENSE
 *
 * add.cu
 * Copyright (C) 2020  Matthew Hardenburgh
 * mdhardenburgh@protonmail.com
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see https://www.gnu.org/licenses/.
 * 
 */
#include <iostream>
#include <math.h>
#include <cuda_profiler_api.h>
#include <string>
#include <stdio.h>
#include <cstdint>

// CUDA Kernel function to add the elements of two arrays on the GPU
//These __global__ functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.
//specifier __global__ to the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.
/**
 * Remember that all of this is happening in parallel. So for THIS SPECIFIC 
 * thread, if the grid increment isnt large enough, it will loop a few more
 * times.
 */
__global__ 
void add(int n, float *x, float *y, uint32_t* threadsInGrid)
{
  uint32_t currentThreadId = threadIdx.x + (blockDim.x * blockIdx.x);
  uint32_t numThreadsInGrid = blockDim.x * gridDim.x;
  *threadsInGrid = numThreadsInGrid;
  uint32_t i = currentThreadId;

  printf("numThreadsInGrid: %d \n", numThreadsInGrid);
  printf("currentThreadId: %d \n", currentThreadId);

  for (uint32_t i = currentThreadId; i < n; i += numThreadsInGrid)
  {
    y[i] = x[i] + y[i];
  }

  // y[i] = x[i] + y[i];
      
}

int main(void)
{
  uint32_t N = 1<<20; // 1M elements
  // uint32_t N = 1064123;

  //To allocate data in unified memory, call cudaMallocManaged()
  //To free the data, just pass the pointer to cudaFree()
  float* x;
  float* y;
  uint32_t* threadsInGrid;

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&threadsInGrid, sizeof(uint32_t));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  uint32_t blockSize = 256;
  uint32_t numBlocks = (N + blockSize - 1)/blockSize;

  // Run kernel on 1M elements on the CPU
  //This is called the execution configuration
  std::cout<<"numBlocks: "<<numBlocks<<std::endl;
  std::cout<<"Number of total calculated theads: "<<numBlocks*blockSize<<std::endl;
  add<<<numBlocks, blockSize>>>(N, x, y, threadsInGrid);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  std::cout<<"Number of theads in grid: "<<*threadsInGrid<<std::endl;

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  // To free the data, just pass the pointer to cudaFree()
  cudaFree(x);
  cudaFree(y);
  cudaFree(threadsInGrid);

  return 0;
}