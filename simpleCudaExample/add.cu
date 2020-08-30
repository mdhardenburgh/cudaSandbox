#include <iostream>
#include <math.h>

// // function to add the elements of two arrays
// void add(int n, float *x, float *y)
// {
//   for (int i = 0; i < n; i++)
//       y[i] = x[i] + y[i];
// }

// CUDA Kernel function to add the elements of two arrays on the GPU
//These __global__ functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.
//specifier __global__ to the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.
__global__ 
void add(int n, float *x, float *y)
{
  int currentThreadId = threadIdx.x;
  int numThreadsInBlock = blockDim.x;
  
  for (int i = currentThreadId; i < n; i += numThreadsInBlock)
  {
    y[i] = x[i] + y[i];
  }
      
}

int main(void)
{
  int N = 1<<20; // 1M elements

  //To allocate data in unified memory, call cudaMallocManaged()
  //To free the data, just pass the pointer to cudaFree()
  float* x;
  float* y;

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  //This is called the execution configuration
  add<<<1, 256>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  // To free the data, just pass the pointer to cudaFree()
  cudaFree(x);
  cudaFree(y);

  return 0;
}