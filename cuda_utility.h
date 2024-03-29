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
#include <omp.h>
#include "utility.h"

#define checkCudaErrors(result) { checkCuda(result, __FILE__, __LINE__); }

enum FILTER_TYPE{
    SAD,
    ZMNCC
};

using namespace std;

inline void checkCuda(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        const char *errorName = cudaGetErrorName(result);
        const char *errorString = cudaGetErrorString(result);
        cerr << "CUDA error (" << errorName << "): " << errorString << " at " << file << ":" << line << endl;
        exit(-1);
    }
}

__global__ void calculate_means_windowed(const float* values, float* means, const int window_size, const int series_length) {
    int start_window_idx = blockIdx.x * blockDim.x + threadIdx.x; // The real thread idx
    int central_idx = start_window_idx + (window_size / 2); // Index of the central value of window
    
    // Only the threads that work with a window that completely covers the data can compute 
    if (central_idx < series_length - window_size / 2){
        float sum = 0.0f;
        for (int i = 0; i <= window_size; ++i){ 
            sum += values[start_window_idx + i]; 
        }
        means[central_idx] = sum / window_size;
    }
}

__global__ void calculate_stds_zmnccs_windowed(const float* values, const float* means, float* stds, float* zmnccs, float* filters, float* filter_means, float* filter_stds, const int window_size, const int series_length) {
    const int start_window_idx = blockIdx.x * blockDim.x + threadIdx.x; // The real thread idx
    const int central_idx = start_window_idx + (window_size / 2); // Index of the central value of window
    
    // Only the threads that work with a window that completely covers the data can compute 
    if (central_idx < series_length - window_size / 2){
        // Compute std
        float variance_summation = 0.0f;
        for (int i = 0; i <= window_size; ++i){
            variance_summation += pow(values[start_window_idx + i] - means[central_idx], 2);
        }
        stds[central_idx] = sqrt(variance_summation / (window_size - 1));

        // Compute zncc
        float zmncc_sum[N_FILTERS];
        memset(zmncc_sum, 0, N_FILTERS * sizeof(float));
        float mean_value = means[central_idx];

        for (int i = 0; i <= window_size; ++i){
            const float current_value = values[start_window_idx + i];
            const float cur_min_mean = current_value - mean_value;
            // Apply more than one filter at the same time
            for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                zmncc_sum[filter_idx] += (cur_min_mean) * (filters[i + filter_idx * FILTER_LENGTH] - filter_means[filter_idx]);
            }
        }

        for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
            zmnccs[central_idx + filter_idx * series_length] = (zmncc_sum[filter_idx] / (window_size * stds[central_idx] * filter_stds[filter_idx]));
        }
    }
}

__global__ void calculate_stds_zmnccs_windowed_shared(const float* values, const float* means, float* stds, float* zmnccs, float* filters, float* filter_means, float* filter_stds, const int window_size, const int series_length, const int values_to_load_per_thread, const int threads_for_loading) {
    extern __shared__ float tile_shared[];

    const int start_window_idx = blockIdx.x * blockDim.x + threadIdx.x; // The real thread idx
    const int central_idx = start_window_idx + (window_size / 2); // Index of the central value of window
    
    const int start_load_idx = start_window_idx * values_to_load_per_thread;

    if(start_window_idx < threads_for_loading){
        // 1)Each thread needs to load the values in its tile
        for(int idx = 0; idx < values_to_load_per_thread; idx++){
            tile_shared[idx] = values[start_load_idx + idx];
        }
    }
    
    // 2) Wait all threads in a block to finish loading
    __syncthreads();

    // 3) All data values needed to compute are loaded, so we can start to calculate

    // Only the threads that work with a window that completely covers the data can compute 
    if (central_idx < series_length - window_size / 2){
        // Compute std
        float variance_summation = 0.0f;
        for (int i = 0; i <= window_size; ++i){
            variance_summation += pow(tile_shared[threadIdx.x + i] - means[central_idx], 2);
        }
        stds[central_idx] = sqrt(variance_summation / (window_size - 1));

        // Compute zncc
        float zmncc_sum[N_FILTERS];
        memset(zmncc_sum, 0, N_FILTERS * sizeof(float));
        float mean_value = means[central_idx];

        for (int i = 0; i <= window_size; ++i){
            const float current_value = tile_shared[threadIdx.x + i];
            const float cur_min_mean = current_value - mean_value;
            // Apply more than one filter at the same time
            for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                zmncc_sum[filter_idx] += (cur_min_mean) * (filters[i + filter_idx * FILTER_LENGTH] - filter_means[filter_idx]);
            }
        }

        for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
            zmnccs[central_idx + filter_idx * series_length] = (zmncc_sum[filter_idx] / (window_size * stds[central_idx] * filter_stds[filter_idx]));
        }
    }
}

__global__ void calculate_SADs(const float* values, float* SADs, float* filters, const int window_size, const int series_length) {
    int start_window_idx = blockIdx.x * blockDim.x + threadIdx.x; // The real thread idx
    int central_idx = start_window_idx + (window_size / 2); // Index of the central value of window
 
    // Only the threads that work with a window that completely covers the data can compute 
    if (central_idx < series_length - window_size / 2){

        float SAD[N_FILTERS];
        memset(SAD, 0, N_FILTERS * sizeof(float));

        for (int i = 0; i <= window_size; ++i) {
            const float current_value = values[start_window_idx + i];
            // Apply more than one filter at the same time
            for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                // Save result in local memory
                SAD[filter_idx] += abs(current_value - filters[i + filter_idx * FILTER_LENGTH]);
            }
        }
        
        // Copy results in global memory
        for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
            SADs[central_idx + filter_idx * series_length] = SAD[filter_idx] / window_size;
        }
    }
}

#endif //TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H