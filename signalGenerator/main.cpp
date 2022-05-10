#include "dataGenerator.h"
#include <math.h>
#include <cstdint>

int main(void)
{
    DataGenerator* myGenerator = new DataGenerator(1000000u);

    float phase = 0;
    float radianFrequency = M_PI/34;
    float amplitude = 1;

    uint32_t blockSize = 256;

    myGenerator->generateWaveform(waveformType::sine, phase, radianFrequency, amplitude, 0, blockSize);

    delete myGenerator;

    return 0;
}
