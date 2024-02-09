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

#define SERIES_LENGTH 101000

using namespace std;

// Function to calculate the mean of a vector of numbers within a range
float calculate_mean_in_range(const vector<float>& values, int start, int end);

// Function to calculate the standard deviation of a vector of numbers within a range
float calculate_standard_deviation_in_range(const vector<float>& values, int start, int end, const float& mean_in_range);


// Print filters
void print_filters(const vector<vector<float>> filters);

#endif //TIME_SERIES_PATTERN_RECOGNITION_UTILITY_H