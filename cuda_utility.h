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
#include <map>

#define SERIES_LENGTH 2075260
#define N_FILTERS 4
#define FILTER_LENGTH 10080
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

__global__ void calculate_stds_znccs_windowed(const float* values, const float* means, float* stds, float* znccs, float* filter, float* filter_mean, float* filter_std, int window_size, int series_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Only the threads that work with a window that completely covers the data can compute 
    if ((idx >= window_size / 2) && (idx < series_length - window_size / 2)) {
        // Compute std
        float variance_summation = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            variance_summation += pow(values[i] - means[idx], 2);
        }
        stds[idx] = sqrt(variance_summation / (window_size - 1));

        // Compute zncc
        float zncc_sum = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            zncc_sum += (values[i] - means[idx]) * (filter[i - idx + window_size / 2] - (*filter_mean));
        }
        znccs[idx] = (zncc_sum / (window_size * stds[idx] * (*filter_std)));
    }
}

__global__ void calculate_SADs(const float* values, float* SADs, float* filter, int window_size, int series_length, int filter_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + (window_size / 2);
    // Only the threads that work with a window that completely covers the data can compute 
    if (idx < series_length - window_size / 2){
        float SAD = 0;     
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            SAD += abs(values[i] - filter[i - idx + window_size / 2]);
        }
        SADs[series_length * filter_idx + idx] = SAD / window_size;
    }
}

vector<float> create_filter_trend_n_weeks(const int n_of_weeks, bool uptrend=true){
    vector<float>filt;
    filt.reserve(n_of_weeks * 1440 * 7);
    if(uptrend){
        float val = 1.0f;
        for(int i = 0; i < n_of_weeks * 1440 * 7; i++){
            filt.push_back(val);
            val++;
        }
    }
    else{
        float val = -1.0f;
        for(int i = 0; i < n_of_weeks * 1440 * 7; i++){
            filt.push_back(val);
            val--;
        }
    }

    return filt;
}


vector<float> create_filter_cycle_n_weeks(const int n_of_weeks, bool up_then_down){

    if((n_of_weeks % 2) != 0){
    cerr << "Number of weeks must be divisible by two" << endl;
    // Return an empty vector to indicate the error
    return vector<float>();
    }

    vector<float>filt;
    filt.reserve(n_of_weeks * 1440 * 7);

    int middle_point = (n_of_weeks / 2) * 1440 * 7;
    if(up_then_down){
        // Add the first half of upgoing values
        float val = 1.0f;
        for(int i = 0; i < middle_point; i++){
            filt.push_back(val);
            val++;
        }
        // Add the second half of downgoing values
        val--;
        for(int i = middle_point; i < n_of_weeks * 1440 * 7; i++){
            filt.push_back(val);
            val--;
        }
    }
    else{
        // Add the first half of upgoing values
        float val = -1.0f;
        for(int i = 0; i < middle_point; i++){
            filt.push_back(val);
            val--;
        }
        // Add the second half of downgoing values
        val++;
        for(int i = middle_point; i < n_of_weeks * 1440 * 7; i++){
            filt.push_back(val);
            val++;
        }
    }

    return filt;
}

#endif //TIME_SERIES_PATTERN_RECOGNITION_CUDA_UTILITY_H