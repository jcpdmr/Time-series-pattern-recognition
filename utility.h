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

#define SERIES_LENGTH 1300

using namespace std;

// Function to calculate the mean of a vector of numbers within a range
float calculateMeanInRange(const vector<float>& values, int start, int end);

// Function to calculate the standard deviation of a vector of numbers within a range
float calculateStandardDeviationInRange(const vector<float>& values, int start, int end);

float calculateCorrelationCoefficient(const vector<float>& data_values, const vector<float>& query_values);

#endif //TIME_SERIES_PATTERN_RECOGNITION_UTILITY_H