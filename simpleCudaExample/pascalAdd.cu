/**                                                                             
 * @file maxwellAdd.cu                                                          
 * @brief simple cuda add for maxwell GPUs and older.                           
 * @author Matthew Hardenburgh                                                  
 * @date 9/13/2020                                                              
 * @copyright Matthew Hardenburgh 2020. All Rights Reserved.                    
 *                                                                              
 * @section license LICENSE                                                                                                                                                                    
 *                                                                                                                                                                                             
 * maxwellAdd.cu                                                                                                                                                                               
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
// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__global__ void init(int n, float *x, float *y) 
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

   // Prefetch the data to the GPU
  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
  cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  init<<<numBlocks, blockSize>>>(N, x, y);
  // Run kernel on 1M elements on the GPU
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
