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

    vector<vector<float>> temp_filter = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f}};
    // temp_filter.push_back(create_filter_trend_n_weeks(1, true));
    // temp_filter.push_back(create_filter_trend_n_weeks(4, true));
    // temp_filter.push_back(create_filter_trend_n_weeks(8, true));

    // temp_filter.push_back(create_filter_trend_n_weeks(1, false));
    // temp_filter.push_back(create_filter_trend_n_weeks(4, false));
    // temp_filter.push_back(create_filter_trend_n_weeks(8, false));

    // temp_filter.push_back(create_filter_cycle_n_weeks(4, true));
    // temp_filter.push_back(create_filter_cycle_n_weeks(8, true));


    // temp_filter.push_back(create_filter_cycle_n_weeks(4, false));
    // temp_filter.push_back(create_filter_cycle_n_weeks(8, false));


    // Create a bank of filters
    const vector<vector<float>> filters = temp_filter;

    auto start_benchmark = chrono::high_resolution_clock::now();

    // #pragma omp parallel for shared(cerr, filters, cout, values) default(none)
    for (int query_idx = 0; query_idx < filters.size(); query_idx++){

        const vector<float> query = filters[query_idx];
        const float query_mean = calculate_mean_in_range(query, 0, query.size());
        const float query_std = calculate_standard_deviation_in_range(query, 0, query.size(), query_mean);
        cout << "Query [" << setw(2) << setfill('0') << query_idx << "] ---> mean: " << query_mean << "    std: " << query_std << endl;

        // At the first iteration we need to put the central query value in a way that 
        // all the filter is "inside" the timeseries. For this reason we don't start
        // from i = 0, but i = QUERY_LENGTH/2
        int offset = query.size() / 2;

        vector<float> zero_norm_cross_correlation_results;
        zero_norm_cross_correlation_results.reserve(SERIES_LENGTH);
        
        for (int i = offset; i <= values.size() - offset; ++i) {
            const float window_values_mean = calculate_mean_in_range(values, i - offset, i + offset + 1);
            const float window_values_std = calculate_standard_deviation_in_range(values, i - offset, i + offset + 1, window_values_mean);

            float zn_cc_sum = 0.0;
            for (size_t j = i - offset; j < i + offset + 1 ; ++j) {
                zn_cc_sum += (values[j] - window_values_mean) * (query[j - i + offset] - query_mean);
            }

            float zn_corr = zn_cc_sum / (query.size() * window_values_std * query_std);
            // if ((i < offset + 5) || (i > SERIES_LENGTH - offset - 5)){
            //     cout << "[" << setw(4) << setfill('0') << i << "] mean: " << fixed << setprecision(5) << window_values_mean << "  std: " << fixed << setprecision(5) <<  window_values_std << "  zn_cc_sum: " << fixed << setprecision(5) << zn_cc_sum << "  zn_corr: " << fixed << setprecision(8) << zn_corr << endl;
            // }
            zero_norm_cross_correlation_results.push_back(zn_corr);

        }

        // Save data
        string file_name = "../output_data/zn_cross_correlation"+ to_string(query_idx) + "_filterlen" + to_string(query.size()) +".txt";
        ofstream output_file(file_name);
        if (output_file.is_open()) {
            for (float value : zero_norm_cross_correlation_results) {
                output_file << value << "\n";
            }
            output_file.close();
            cout << file_name << endl;
        } else {
            cerr << file_name << endl;
        }
    }

    auto stop_benchmark = chrono::high_resolution_clock::now();
    auto duration_benchmark = chrono::duration_cast<chrono::milliseconds >(stop_benchmark - start_benchmark).count();
    
    cout << "Benchmark elapsed time: " << duration_benchmark << " ms" << endl;
    return 0;
}
