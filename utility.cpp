#include "utility.h"

void calculate_means_windowed(const vector<float>& values, map<int,vector<float>>& means, int window_size, int series_length) {
    for(int idx = window_size / 2; idx < series_length - window_size / 2; idx++){
        float sum = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            sum += values[i];
        }
        means[window_size][idx] = sum / window_size;
    }
}

void calculate_stds_windowed(const vector<float>& values, const map<int,vector<float>>& means, map<int,vector<float>>& stds, int window_size, int series_length) {
    for(int idx = window_size / 2; idx < series_length - window_size / 2; idx++){
        float variance_summation = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            variance_summation += pow(values[i] - (means.find(window_size)->second[idx]), 2);
        }
        stds[window_size][idx] = sqrt(variance_summation / (window_size - 1)) ;
    }
}

void calculate_znccs_windowed(const vector<float>& values, const map<int,vector<float>>& means, const map<int,vector<float>>& stds, map<int, vector<float>>& zncc, const vector<float>& filter, float filter_mean, float filter_std, int window_size, int series_length) {
    for(int idx = window_size / 2; idx < series_length - window_size / 2; idx++){
        float zncc_sum = 0.0f;
        for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
            zncc_sum += (values[i] - (means.find(window_size)->second[idx])) * (filter[i - idx + window_size / 2] - filter_mean);
            // if (idx == 5){
            //     printf("[%i] Val: %f  FVal: %f\n", i, values[i], filter[i - idx + window_size / 2]);
            //     printf("[%i] V - M: %f   F - M: %f   Curr: %f   Sum: %f \n", i, (values[i] - means[idx]), (filter[i - idx + window_size / 2] - filter_mean) ,(values[i] - means[idx]) * (filter[i - idx + window_size / 2] - filter_mean), zncc_sum);
            // }
        }
        zncc[window_size][idx] = (zncc_sum / (window_size * (stds.find(window_size)->second[idx]) * filter_std));
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