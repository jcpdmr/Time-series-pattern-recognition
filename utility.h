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

#define SERIES_LENGTH 2100000

using namespace std;

// Calculate the mean of a vector of numbers within a range
float calculate_mean_in_range(const vector<float>& values, int start, int end);

// Calculate the standard deviation of a vector of numbers within a range
float calculate_standard_deviation_in_range(const vector<float>& values, int start, int end, const float& mean_in_range);

// Create a trend filter for n weeks, uptrend true if filter is and uptrend, false if downtrend
vector<float> create_filter_trend_n_weeks(const int n_of_weeks, bool uptrend);

// Create a cycle filter for n weeks, up_then_down true if first half of weeks uptrend an other half downtrend, false if 
// first half of weeks downtrend an other half uptrend
vector<float> create_filter_cycle_n_weeks(const int n_of_weeks, bool up_then_down);


// Print filters
void print_filters(const vector<vector<float>>& filters);

#endif //TIME_SERIES_PATTERN_RECOGNITION_UTILITY_H