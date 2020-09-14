# Simple CUDA Example
This simple project is to get my feet wet with CUDA programming. CUDA toolkit 
install instructions are in the main directory. This example is an 
implementation of the 2017 CUDA blogpost [An Even Easier Introduction to CUDA.](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
with a few modifications to work on the Jetson Nano.

## Some CUDA Language Explained
* Host and Device. Host is the CPU and is used to refer to code running on the 
  CPU. Device is the CUDA processor and is used to rever to code running on the GPU.
* `__global__ ` Defines a function that can be called from host and executed on 
  the GPU.
* `__device__` Defines a function that can be called by global function. Device
  function can only be called from the GPU.
* `add<<<numBlocks, blockSize>>>(...);` Sample of CUDA 
  kernel call semantics. The first argument `numBlocks` is the number of blocks 
  in a grid and the second argument `blockSize` defines the number of threads
  in a block, AKA the size of the block. It should be noted that CUDA GPUs run 
  kernels using blocks of threads that are a multiple of 32.
* `cudaMallocManaged()` Allocates memory in the unified memory space so that it 
  can be accessed by both the CPU and GPU. Similar to `new`.
* `cudaDeviceSynchronize()` Tells the CPU to wait for the GPU computation to
  finish. CUDA kernel launches don't block the CPU. 
* `cudaFree()` Frees the allocated unified memory. Similar to `delete`.
* `threadIdx.x` Is the ID of the current thread
* `blockDim.x`  Is the number of threads in the current block
* `blockIdx.x` Is the ID of the current block
* `gridDim.x` Is number of blocks in the current grid

### CUDA Indexing. Source Nvida, "An Even Easier Introduction to CUDA"
![CUDA Indexing. Source Nvida, "An Even Easier Introduction to CUDA"](cuda_indexing.png)

## Grid Stride Loops Explained
A tool that is introduced in this example without much explanation is the use of 
so called "grid-stride loops".

Logically we would want to launch one thread per data element, under the 
assumption we have enough threads to cover the array. However, this assumption
may not be true all the time for all situations. Instead of assuming the 
thread grid is large enough to cover the size of the array, the CUDA kernel 
loops over the array one grid-size at a time. A grid is made up of blocks and
blocks are made up of threads. Therefore the total number of threads in a grid,
and thus the grid-size is `blockDim.x * gridDim.x`.

When a kernel is launched with a defined grid-size large enough to cover the
whole loop, then the loop is ignored and all operations happen in parallel. 

### Why Would We Do This?
1. By using a loop, any size of data can be tackled. Even if it exceeds the
   largest supported grid-size. 
2. Performance reasons. It is advised to launch grids that have the same
   number of blocks that is a multiple of the number of multiprocessor on the
   device.
3. Debugging. A parallel program can be much easier converted back to a serial
   program when debugging. 
4. Writing CUDA device portable code.

[More information on grid-stride loops](https://developer.nvidia.com/blog/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)

## How to Use This Example:

To compile this example:
`$ sudo /usr/local/cuda-11.0/bin/nvcc add.cu -g -o cudaAdd` 

To cross compile for the Jetson Nano:
`$ sudo /usr/local/cuda-10.2/bin/nvcc add.cu -g -o armCudaAdd --compiler-bindir ~/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++`

To run it:
`$ ./cudaAdd`
or
`$ ./armCudaAdd`

To profile it:
`$ nvprof ./cudaAdd`
or
`$ /usr/local/cuda-10.2/bin/nvprof ./armCudaAdd`

## Simple Speed Metrics
Time to run on my laptop's Nvidia 960m 175.81us
Time to run on my Jetson Nano is 2.561ms
