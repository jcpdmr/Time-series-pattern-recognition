#include "utility.h"

float calculate_mean_in_range(const vector<float>& values, int start, int end) {
    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        sum += values[i];
    }
    return sum / (end - start);
}

float calculate_standard_deviation_in_range(const vector<float>& values, int start, int end, const float& mean_in_range) {
    float variance = 0.0f;
    for (int i = start; i < end; ++i) {
        variance += pow(values[i] - mean_in_range, 2);
    }
    variance /= (end - start);
    return sqrt(variance);
}

void print_filters(const vector<vector<float>> filters){
    for (const vector<float> query : filters){
        cout << "Query: ";
        for(float value : query){
            cout << value << ", ";
        }
        cout << endl;
    }
}