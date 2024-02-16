#include "utility.h"

int main() {
    // Open the file
    ifstream file("../input_data/household_power_consumption_short.txt");
    if (!file.is_open()) {
        cerr << "Error opening the file!" << endl;
        return 1;
    }

    vector<string> dates;
    vector<float> values;
	dates.reserve(SERIES_LENGTH);
	values.reserve(SERIES_LENGTH);

    string line;

	// Skip first line (because it contains the header row)
	getline(file, line);

    while (getline(file, line)) {
        istringstream iss(line);
        string token;

        // Read the values of Date and Time columns
        getline(iss, token, ';'); // Date
        string date = token;
        getline(iss, token, ';'); // Time
        date += " " + token; // Combine Date and Time

        // Read the value of the Global_active_power column
        getline(iss, token, ';');
        float power;
        // In case of missing data (symbol "?") power gets 0
        istringstream(token) >> power;
        
        // Add the values to the vectors
        dates.push_back(date);
        values.push_back(power);
    }

    // Remeber to close the file
    file.close();

    vector<vector<float>> temp_filter;
    temp_filter.push_back(create_filter_trend_n_weeks(1, true));
    temp_filter.push_back(create_filter_trend_n_weeks(4, true));
    temp_filter.push_back(create_filter_trend_n_weeks(8, true));

    temp_filter.push_back(create_filter_trend_n_weeks(1, false));
    temp_filter.push_back(create_filter_trend_n_weeks(4, false));
    temp_filter.push_back(create_filter_trend_n_weeks(8, false));

    temp_filter.push_back(create_filter_cycle_n_weeks(4, true));
    temp_filter.push_back(create_filter_cycle_n_weeks(8, true));


    temp_filter.push_back(create_filter_cycle_n_weeks(4, false));
    temp_filter.push_back(create_filter_cycle_n_weeks(8, false));


    // Create a bank of filters
    const vector<vector<float>> filters = temp_filter;

    cout << "Executing benchmark..." << endl;
    auto start_benchmark = chrono::high_resolution_clock::now();

    // vector<vector<float>> SADs(filters.size());
    
    // for (int query_idx = 0; query_idx < filters.size(); query_idx++){
    //     const vector<float> query = filters[query_idx];
    //     const int window_size = query.size();
    //     SADs[query_idx] = vector<float>(SERIES_LENGTH, 0.0f);
    //     #pragma omp parallel for
    //     for(int idx = window_size / 2; idx < SERIES_LENGTH - window_size / 2; idx++){
    //         float SAD = 0;     
    //         for (int i = idx - window_size / 2; i <= idx + window_size / 2; ++i) {
    //             SAD += abs(values[i] - query[i - idx + window_size / 2]);
    //         }
    //         SADs[query_idx][idx] = SAD / window_size;
    //     }
    // }

    // Save data
    #pragma omp parallel for
    for(int query_idx = 0; query_idx < filters.size(); query_idx++){
        const vector<float> query = filters[query_idx];

        string file_name = "../output_data/SAD"+ to_string(query_idx) + "_filterlen" + to_string(query.size()) +".txt";
        ofstream output_file(file_name);
        if (output_file.is_open()) {
            for (float value : SADs[query_idx]) {
                output_file << std::fixed << std::setprecision(5) << value << "\n";
            }
            output_file.close();
            cout << "Saved successfully: " << file_name << endl;
        } else {
            cerr << "Failed to open: " << file_name << endl;
        }
    }

    map<int, vector<float>> means, stds; // int is the filter length, vector<float> the mean or std value
    map<int, pair<float, float>> filters_stats; // int is the filter idx, pair<float, float> first is mean, second is std
    map<int, vector<float>> znccs;

    #pragma omp parallel
    {
        
        // Pre-compute all means and stds for the time series data using window size of all filters, also
        // pre-compute the mean and std of all filter
        #pragma omp for
        for (int query_idx = 0; query_idx < filters.size(); query_idx++){
            const vector<float> query = filters[query_idx];
            const int window_size = query.size();

            means[window_size] = vector<float>(SERIES_LENGTH, 0.0f);
            calculate_means_windowed(values, means, window_size, SERIES_LENGTH);

            stds[window_size] = vector<float>(SERIES_LENGTH, 0.0f);
            calculate_stds_windowed(values, means, stds, window_size, SERIES_LENGTH);

            float sum = 0.0f;
            for (int i = 0; i <= window_size; ++i) {
                sum += query[i];
            }
            filters_stats[query_idx].first = sum / window_size;

            float variance_summation = 0.0f;
            for (int i = 0; i < window_size; ++i) {
                variance_summation += pow(query[i] - (filters_stats[query_idx].first), 2);
            }
            filters_stats[query_idx].second = sqrt(variance_summation / (window_size - 1));
        }

        #pragma omp barrier
    
        // Calculate znccs for all filters
        #pragma omp for
        for (int query_idx = 0; query_idx < filters.size(); query_idx++){

            const vector<float> query = filters[query_idx];
            const int window_size = query.size();

            znccs[window_size] = vector<float>(SERIES_LENGTH, 0.0f); 
            calculate_znccs_windowed(values, means, stds, znccs, query, filters_stats[query_idx].first, filters_stats[query_idx].second, window_size, SERIES_LENGTH);
        }

        #pragma omp barrier

        // Save data
        #pragma omp for
        for(int query_idx = 0; query_idx < filters.size(); query_idx++){
            const vector<float> query = filters[query_idx];
            const int window_size = query.size();

            string file_name = "../output_data/zn_cross_correlation"+ to_string(query_idx) + "_filterlen" + to_string(query.size()) +".txt";
            ofstream output_file(file_name);
            if (output_file.is_open()) {
                for (float value : znccs[window_size]) {
                    output_file << value << "\n";
                }
                output_file.close();
                cout << "Saved successfully: " << file_name << endl;
            } else {
                cerr << "Failed to open: " << file_name << endl;
            }
        }
        
    }

    auto stop_benchmark = chrono::high_resolution_clock::now();
    auto duration_benchmark = chrono::duration_cast<chrono::milliseconds >(stop_benchmark - start_benchmark).count();
    
    cout << "Benchmark elapsed time: " << duration_benchmark << " ms" << endl;
    return 0;
}
