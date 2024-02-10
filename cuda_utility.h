#ifndef TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H
#define TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H

#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx);
}

#endif //TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H