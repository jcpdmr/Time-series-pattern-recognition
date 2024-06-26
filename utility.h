#ifndef TIME_SERIES_PATTERN_RECOGNITION_UTILITY_H
#define TIME_SERIES_PATTERN_RECOGNITION_UTILITY_H

#include <omp.h>
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
#include <filesystem>
#include <random>

using namespace std;
namespace fs = std::filesystem;

#define SERIES_LENGTH 2075260 * 1
#define N_FILTERS 8
#define FILTER_LENGTH 12032

#define N_TEST 8 // Number of test to execute

#define BLOCK_SIZE 64 // Dimension of block used in CUDA

string CPU_flt_type = "ZMNCC";  // "SAD" or "ZMNCC"

string GPU_flt_type = "ZMNCC_CUDA"; // "SAD_CUDA" or "ZMNCC_CUDA"
string GPU_use_shared_mem = "NO"; // "SHARED": use shared memory, "NO": no shared memory

bool save_data = false; // Decide if save computing result SAD and ZMNCC or skip save

void calculate_means_windowed(const vector<float>& values, vector<float>& means, const int window_size, const int series_length){
    #pragma omp parallel for
    for(int central_idx = window_size / 2; central_idx < series_length - window_size; central_idx++){
        int start_window_idx = central_idx - window_size / 2;
        float sum = 0.0f;
        for (int i = 0; i <= window_size; ++i){ 
            sum += values[start_window_idx + i]; 
        }
        means[central_idx] = sum / window_size;
    }
};

void calculate_stds_zmnccs_windowed(const vector<float>& values, const vector<float>& means,  vector<float>& stds, vector<float>& zmnccs, vector<float>& filters, vector<float>& filter_means, vector<float>& filter_stds, const int window_size, const int series_length){
    #pragma omp parallel for
    for(int central_idx = window_size / 2; central_idx < series_length - window_size; central_idx++){
        int start_window_idx = central_idx - window_size / 2;
        // Compute std
        float variance_summation = 0.0f;
        for (int i = 0; i <= window_size; ++i){
            variance_summation += pow(values[start_window_idx + i] - means[central_idx], 2);
        }
        stds[central_idx] = sqrt(variance_summation / (window_size - 1));
        
        // Compute zncc
        vector<float> zmncc_sum(N_FILTERS, 0.0f);
        float mean_value = means[central_idx];

        for (int i = 0; i <= window_size; ++i){
            const float current_value = values[start_window_idx + i];
            const float cur_min_mean = current_value - mean_value;
            // Apply more than one filter at the same time
            for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                zmncc_sum[filter_idx] += (cur_min_mean) * (filters[i + filter_idx * window_size] - filter_means[filter_idx]);
            }
        }

        for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
            zmnccs[central_idx + filter_idx * series_length] = (zmncc_sum[filter_idx] / (window_size * stds[central_idx] * filter_stds[filter_idx]));
        }
    }
};

// Create a filter of floats with lenght len. The values of filter are randomly generated
vector<float> create_filter(const int len){
    vector<float>filt;
    filt.reserve(len);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distribution(1,1000);
    for(int i = 0; i < len; i++){
        int rnd_num = distribution(gen);
        filt.push_back(rnd_num);
    }

    return filt;
}

// Create a filter of floats with lenght len. The values of filter increase with step of 1 unit
vector<float> create_filter_trend(const int len, bool uptrend=true){
    vector<float>filt;
    filt.reserve(len);
    if(uptrend){
        float val = 1.0f;
        for(int i = 0; i < len ; i++){
            filt.push_back(val);
            val++;
        }
    }
    else{
        float val = -1.0f;
        for(int i = 0; i < len; i++){
            filt.push_back(val);
            val--;
        }
    }

    return filt;
}

#endif //TIME_SERIES_PATTERN_RECOGNITION_UTILITY_H