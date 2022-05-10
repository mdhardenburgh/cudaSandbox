#include "dataGenerator.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

DataGenerator::DataGenerator()
{

}

DataGenerator::DataGenerator(uint32_t numSamples)
{
    (*this).numSamples = numSamples;
}

DataGenerator::~DataGenerator()
{

}

void DataGenerator::generateWaveform(waveformType wfm, float phase, float frequency, float amplitude, uint32_t noise, uint32_t blockSize)
{
    float* y_0;
    cudaMallocManaged(&y_0, numSamples*sizeof(float));
    (void)noise; // stubb this out for later 
    std::string fileName;
    uint32_t numBlocks = (numSamples + blockSize - 1)/blockSize;

    if(wfm == waveformType::sine)
    {
        launchSineKernel(y_0, phase, frequency, amplitude, numBlocks, blockSize);
        fileName = "sineData.out";
    }

    writeOutToFile(y_0, fileName);
    cudaFree(&y_0);
    return; 
}

void DataGenerator::writeOutToFile(float* y_0, std::string fileName)
{
    std::fstream myFile(fileName, std::fstream::out);
    
    for(uint32_t i = 0; i < numSamples; i++)
    {
        myFile << std::fixed << std::setprecision(10) << y_0[i] << std::endl;
    }
    myFile.close();
}
