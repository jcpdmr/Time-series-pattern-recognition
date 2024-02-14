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
#include <map>

#define SERIES_LENGTH 525600

using namespace std;

void calculate_means_windowed(const vector<float>& values, map<int,vector<float>>& means, int window_size, int series_length);
void calculate_stds_windowed(const vector<float>& values, const map<int,vector<float>>& means, map<int,vector<float>>& stds, int window_size, int series_length);
void calculate_znccs_windowed(const vector<float>& values, const map<int,vector<float>>& means, const map<int,vector<float>>& stds, map<int, vector<float>>& zncc, const vector<float>& filter, float filter_mean, float filter_std, int window_size, int series_length);

// Create a trend filter for n weeks, uptrend true if filter is and uptrend, false if downtrend
vector<float> create_filter_trend_n_weeks(const int n_of_weeks, bool uptrend);

// Create a cycle filter for n weeks, up_then_down true if first half of weeks uptrend an other half downtrend, false if 
// first half of weeks downtrend an other half uptrend
vector<float> create_filter_cycle_n_weeks(const int n_of_weeks, bool up_then_down);

#endif //TIME_SERIES_PATTERN_RECOGNITION_UTILITY_H