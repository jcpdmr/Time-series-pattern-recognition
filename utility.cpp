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

void print_filters(const vector<vector<float>>& filters){
    for (const vector<float>& query : filters){
        cout << "Query: ";
        for(float value : query){
            cout << value << ", ";
        }
        cout << endl;
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