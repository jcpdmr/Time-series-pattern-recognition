#ifndef TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H
#define TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include <chrono>

#define SERIES_LENGTH 525600

using namespace std;

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(result) << endl;
        exit(-1);
    }
    return result;
}

__global__ void calculate_means_in_range(const float* values, float* means, int window_size, int series_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Only the threads that work with a window that completely covers the data can compute 
    if ((idx > window_size / 2) && (idx <= series_length - window_size / 2)) {
        float sum = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size; ++i) {
            sum += values[i];
        }
        means[idx] = sum / window_size;
    }
}

#endif //TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H