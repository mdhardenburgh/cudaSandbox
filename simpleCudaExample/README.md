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

## Pascal (10 series and above) vs Maxwell Memory Arch
https://developer.nvidia.com/blog/unified-memory-cuda-beginners/

For newer GPUs, to get the performance you expect, you must to some extra steps.

## How to Use This Example:

To compile this example for Pascal architectures and above:
`$ sudo /usr/local/cuda-11.6/bin/nvcc pascalAdd.cu -g -o cudaAdd` 

To cross compile for the Jetson Nano and other Maxwell devices:
`$ sudo /usr/local/cuda-10.2/bin/nvcc maxwellAdd.cu -g -o armCudaAdd --compiler-bindir ~/<l4t-install-dir>/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++`

To run it:
`$ ./cudaAdd`
or
`$ ./armCudaAdd`

To profile it:
`$ /usr/local/cuda-11.6/bin/nvprof ./cudaAdd`
or
`$ /usr/local/cuda-10.2/bin/nvprof ./armCudaAdd`

## Simple Speed Metrics
Time to run on my old Asus laptop Nvidia 960m 175.81us

Time to run on my Jetson Nano is 2.561ms

Time to run on my GTX 1070 is about 104us, full run:

==4241== NVPROF is profiling process 4241, command: ./cudaAddMax error: 0
==4241== Profiling application: ./cudaAdd
==4241== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.43%  64.513us         1  64.513us  64.513us  64.513us  add(int, float*, float*)
                   38.57%  40.513us         1  40.513us  40.513us  40.513us  init(int, float*, float*)
      API calls:   99.26%  152.30ms         2  76.151ms  81.695us  152.22ms  cudaMallocManaged
                    0.27%  418.58us         2  209.29us  38.143us  380.43us  cudaMemPrefetchAsync
                    0.25%  386.99us         2  193.49us  117.62us  269.36us  cudaFree
                    0.09%  138.15us       101  1.3670us     190ns  59.032us  cuDeviceGetAttribute
                    0.06%  88.117us         1  88.117us  88.117us  88.117us  cudaDeviceSynchronize
                    0.04%  65.635us         2  32.817us  13.626us  52.009us  cudaLaunchKernel
                    0.01%  22.533us         1  22.533us  22.533us  22.533us  cuDeviceGetName
                    0.00%  6.9540us         1  6.9540us  6.9540us  6.9540us  cuDeviceGetPCIBusId
                    0.00%  4.0570us         1  4.0570us  4.0570us  4.0570us  cudaGetDevice
                    0.00%  2.2230us         3     741ns     290ns  1.3820us  cuDeviceGetCount
                    0.00%  1.1820us         2     591ns     220ns     962ns  cuDeviceGet
                    0.00%     441ns         1     441ns     441ns     441ns  cuDeviceTotalMem
                    0.00%     350ns         1     350ns     350ns     350ns  cuDeviceGetUuid

==4241== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1070 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  451.4050us  Device To Host
Total CPU Page faults: 12
