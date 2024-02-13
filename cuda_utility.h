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

#define SERIES_LENGTH 52560
#define checkCudaErrors(result) { checkCuda(result, __FILE__, __LINE__); }

using namespace std;

inline void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        const char *errorName = cudaGetErrorName(result);
        const char *errorString = cudaGetErrorString(result);
        cerr << "CUDA error (" << errorName << "): " << errorString << " at " << file << ":" << line << endl;
        exit(-1);
    }
}

__global__ void calculate_means_windowed(const float* values, float* means, int window_size, int series_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Only the threads that work with a window that completely covers the data can compute 
    if ((idx >= window_size / 2) && (idx < series_length - window_size / 2)) {
        float sum = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            sum += values[i];
        }
        means[idx] = sum / window_size;
    }
}

__global__ void calculate_stds_windowed(const float* values, const float* means, float* stds, int window_size, int series_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Only the threads that work with a window that completely covers the data can compute 
    if ((idx >= window_size / 2) && (idx < series_length - window_size / 2)) {
        float variance_summation = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            variance_summation += pow(values[i] - means[idx], 2);
        }
        stds[idx] = sqrt(variance_summation / (window_size - 1)) ;
    }
}

__global__ void calculate_znccs_windowed(const float* values, const float* means, float* stds, float* zncc, float* filter, float filter_mean, float filter_std, int window_size, int series_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Only the threads that work with a window that completely covers the data can compute 
    if ((idx >= window_size / 2) && (idx < series_length - window_size / 2)) {
        float zncc_sum = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            zncc_sum += (values[i] - means[idx]) * (filter[i - idx + window_size / 2] - filter_mean);
            // if (idx == 5){
            //     printf("[%i] Val: %f  FVal: %f\n", i, values[i], filter[i - idx + window_size / 2]);
            //     printf("[%i] V - M: %f   F - M: %f   Curr: %f   Sum: %f \n", i, (values[i] - means[idx]), (filter[i - idx + window_size / 2] - filter_mean) ,(values[i] - means[idx]) * (filter[i - idx + window_size / 2] - filter_mean), zncc_sum);
            // }
        }
        zncc[idx] = (zncc_sum / (window_size * stds[idx] * filter_std));
        if(idx < 10 && idx > 4){
            printf("[%i] std: %f \n", idx, stds[idx]);
            // printf("[%i] zncc: %f", idx, zncc[idx]);
        }
    }
}

#endif //TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H