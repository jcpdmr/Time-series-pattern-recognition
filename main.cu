#include "cuda_utility.h"

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

    float *d_values, *d_means;
    checkCuda(cudaMalloc((void**)&d_values, SERIES_LENGTH * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_means, SERIES_LENGTH * sizeof(float)));

    // Copying dei dati dalla CPU alla GPU
    checkCuda(cudaMemcpy(d_values, values.data(), SERIES_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

    const int window_size = 11;
    const int block_size = 256;
    const int n_blocks = (SERIES_LENGTH + block_size - 1) / block_size;

    calculate_means_in_range<<<n_blocks, block_size>>>(d_values, d_means, window_size, SERIES_LENGTH);
    checkCuda(cudaGetLastError());

    // Wait for the kernel execution to finish
    checkCuda(cudaDeviceSynchronize());

    // Copy results from GPU to CPU
    vector<float> means(SERIES_LENGTH, 0.0f);
    checkCuda(cudaMemcpy(means.data(), d_means, SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Free resources
    cudaFree(d_values);
    cudaFree(d_means);

    // Print results
    for (int i = 0; i < SERIES_LENGTH; ++i) {
        if ((i < window_size + 20) || (i > SERIES_LENGTH - window_size - 20)){
            cout << "Mean at index " << i << ": " << means[i] << endl;
        }
    }

    cout << "Finished" << endl;

    return 0;
}