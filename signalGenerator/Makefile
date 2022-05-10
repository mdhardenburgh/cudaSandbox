CXX = g++
CXX_FLAGS = -g -c -std=c++11 -Wall -W -Werror -pedantic -I/../../usr/local/cuda-11.2/include
NVCC=/usr/local/cuda-11.2/bin/nvcc  
CUDA_FLAGS= -c -arch=sm_50 -Wno-deprecated-gpu-targets -x cu -Werror all-warnings -g -G --compiler-bindir $(CXX)
CUDA_INCLDUES = -I/../../usr/local/cuda-11.2/include

dataGen: main.o dataGenerator.o cudaDataGenerator.o 
	$(CXX) $^ -L/usr/local/cuda-11.2/lib64 -lcudart -lcuda -o $@     

main.o: main.cpp
	$(CXX) $^ $(CXX_FLAGS) $(CUDA_INCLUDES) -o $@ 

dataGenerator.o: dataGenerator.cpp 
	$(CXX) $^ $(CXX_FLAGS) -o $@ 

cudaDataGenerator.o: cudaDataGenerator.cu 
	$(NVCC) $^ $(CUDA_FLAGS) -o $@ 

clean:
	rm -f *.o dataGen
