#include "utility.h"

int main() {
    // Open the file
    ifstream file("../input_data/NVDA.csv");
    if (!file.is_open()) {
        cerr << "Error opening the file!" << endl;
        return 1;
    }

    vector<string> dates;
    vector<float> values;
	dates.reserve(SERIES_LENGTH);
	values.reserve(SERIES_LENGTH);

    string line;

	// Need to skip first line (because it contains the header row)
	getline(file, line);

    while (getline(file, line)) {
        istringstream line_stream(line);
        string token;
        vector<string> tokens;
		tokens.reserve(7 * SERIES_LENGTH);

        while (getline(line_stream, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() >= 6) {
			// Get date and close value
            dates.push_back(tokens[0]);
            values.push_back(stof(tokens[4])); 
        }
    }

    // Remeber to close the file
    file.close();

    const vector<float> uptrend_query = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f};
    const float query_mean = calculateMeanInRange(uptrend_query, 0, uptrend_query.size());
    const float query_std = calculateStandardDeviationInRange(uptrend_query, 0, uptrend_query.size());

    cout << "Query ---> mean: " << query_mean << "    std: " << query_std << endl;

    // At the first iteration we need to put the query central value in a way that 
    // all the filter is "inside" the timeseries. For this reason we don't start
    // from i = 0, but i = QUERY_LENGTH/2 = 3, in questo modo inizieremo dal quarto elemento
    int offset = uptrend_query.size() / 2;

    vector<double> correlation_results;
	correlation_results.reserve(SERIES_LENGTH);
    
    for (int i = offset; i <= values.size() - offset; ++i) {
        float window_values_mean = calculateMeanInRange(values, i - offset, i + offset + 1);
        float window_values_std = calculateStandardDeviationInRange(values, i - offset, i + offset + 1);



        double sum = 0.0f;
        for (size_t j = i - offset; j < i + offset + 1 ; ++j) {
            sum += (values[j] - window_values_mean) * (uptrend_query[j - i + offset] - query_mean);
        }

        double corr = sum / (uptrend_query.size() * window_values_std * query_std);
        cout << "[" << setw(4) << setfill('0') << i << "] mean: " << fixed << setprecision(5) << window_values_mean << "  std: " << fixed << setprecision(5) <<  window_values_std << "  sum: " << fixed << setprecision(5) << sum << "  corr: " << fixed << setprecision(8) << corr << endl;

        correlation_results.push_back(corr);

        // cout << "Correlation coefficient for window " << i << ": " << corr << endl;
    }

    // Save data
    ofstream output_file("../output_data/correlation.csv");
    if (output_file.is_open()) {
        for (float value : correlation_results) {
            output_file << value << "\n";
        }
        output_file.close();
        cout << "Output data written output_data/correlation.csv" << endl;
    } else {
        cerr << "Unable to open output_data/correlation.csv" << endl;
    }

    return 0;
}
