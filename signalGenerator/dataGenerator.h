/**
 * @file dataGenerator.h
 * @brief This class will be used to generate training sets for the DLO.
 * @author Matthew Hardenburgh
 * @date 10/18/2020
 * @copyright Delta Technologies
 */

#include <iostream>
#include <math.h>
#include <string>
#include <stdio.h>
#include <cstdint>
#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <iostream>
#include <fstream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum class waveformType: uint32_t{sine, square, ramp, triangle};

class DataGenerator
{
    public:
        DataGenerator();
        DataGenerator(uint32_t numSamples);
        ~DataGenerator();

        void generateWaveform(waveformType wfm, float phase, float frequency, float amplitude, uint32_t noise, uint32_t blockSize);
    private:
        void launchSineKernel(float* y_0, float phase, float frequency, float amplitude, uint32_t numBlocks, uint32_t blockSize);
        void writeOutToFile(float* y_0, std::string fileName);
        
        uint32_t numSamples = 1e6;
};

#endif
