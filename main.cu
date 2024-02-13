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

    const int window_size = 11;
    const int block_size = 256;
    const int n_blocks = (SERIES_LENGTH + block_size - 1) / block_size;

    const vector<float> filter = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    float *d_values, *d_means, *d_stds, *d_znccs, *d_filter;
    checkCudaErrors(cudaMalloc((void**)&d_values, SERIES_LENGTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_means, SERIES_LENGTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_stds, SERIES_LENGTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_filter, filter.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_znccs, SERIES_LENGTH * sizeof(float)));

    // Copying data from CPU to GPU
    checkCudaErrors(cudaMemcpy(d_values, values.data(), SERIES_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice));

    calculate_means_windowed<<<n_blocks, block_size>>>(d_values, d_means, window_size, SERIES_LENGTH);
    checkCudaErrors(cudaGetLastError());
    // Wait for the kernel execution to finish
    checkCudaErrors(cudaDeviceSynchronize());

    calculate_stds_windowed<<<n_blocks, block_size>>>(d_values, d_means, d_stds, window_size, SERIES_LENGTH);
    checkCudaErrors(cudaGetLastError());
    // Wait for the kernel execution to finish
    checkCudaErrors(cudaDeviceSynchronize());

    calculate_znccs_windowed<<<n_blocks, block_size>>>(d_values, d_means, d_stds, d_znccs, d_filter, 6.0f, 3.3166247903554, window_size, SERIES_LENGTH);
    checkCudaErrors(cudaGetLastError());
    // Wait for the kernel execution to finish
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy results from GPU to CPU
    vector<float> means(SERIES_LENGTH, 0.0f);
    vector<float> stds(SERIES_LENGTH, 0.0f);
    vector<float> znccs(SERIES_LENGTH, 0.0f);
    checkCudaErrors(cudaMemcpy(means.data(), d_means, SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(stds.data(), d_stds, SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(znccs.data(), d_znccs, SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Free resources
    cudaFree(d_values);
    cudaFree(d_means);
    cudaFree(d_stds);
    cudaFree(d_znccs);
    cudaFree(d_filter);

    // Print results
    // for (int i = 0; i < SERIES_LENGTH; ++i) {
    //     if ((i < window_size + 5) || (i > SERIES_LENGTH - window_size - 5)){
    //         cout << "Means at index " << i << ": " << means[i] << endl;
    //     }
    // }
    // for (int i = 0; i < SERIES_LENGTH; ++i) {
    //     if ((i < window_size + 5) || (i > SERIES_LENGTH - window_size - 5)){
    //         cout << "Std at index " << i << ": " << stds[i] << endl;
    //     }
    // }
    // for (int i = 0; i < SERIES_LENGTH; ++i) {
    //     if ((i < window_size + 5) || (i > SERIES_LENGTH - window_size - 5)){
    //         cout << "Zncc at index " << i << ": " << znccs[i] << endl;
    //     }
    // }

    cout << "Finished" << endl;

    return 0;
}