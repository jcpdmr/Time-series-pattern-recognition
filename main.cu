#include "cuda_utility.h"

int main() {
    // Open the file
    ifstream file("../input_data/household_power_consumption.txt");
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

    vector<vector<float>> temp_filters;
    temp_filters.push_back(create_filter_trend_n_weeks(1, true));
    temp_filters.push_back(create_filter_trend_n_weeks(1, true));
    temp_filters.push_back(create_filter_trend_n_weeks(1, true));
    temp_filters.push_back(create_filter_trend_n_weeks(1, true));
    temp_filters.push_back(create_filter_trend_n_weeks(1, true));
    temp_filters.push_back(create_filter_trend_n_weeks(1, true));
    temp_filters.push_back(create_filter_trend_n_weeks(1, true));
    temp_filters.push_back(create_filter_trend_n_weeks(1, true));
    // Create a bank of filters
    const vector<vector<float>> filters = temp_filters;

    auto start_benchmark = chrono::high_resolution_clock::now();
    
    const int block_size = 256;

    // Allocate CPU space for the result
    vector<float> SADs(filters.size() * SERIES_LENGTH, 0.0f);

    // Allocate GPU space for data values, results and filters
    float *d_values, *d_SADs, *d_filters;
    checkCudaErrors(cudaMalloc((void**)&d_values, SERIES_LENGTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_SADs, filters.size() * SERIES_LENGTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_filters, filters.size() * FILTER_LENGTH * sizeof(float)));

    // Copying data values and filters from CPU to GPU
    checkCudaErrors(cudaMemcpy(d_values, values.data(), SERIES_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filters , filters.data(), filters.size() * FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

    for (int query_idx = 0; query_idx < filters.size(); query_idx++){
        const vector<float> query = filters[query_idx];
        const int window_size = query.size();
        
        // checkCudaErrors(cudaMemcpy(d_filters + query_idx * query.size(), query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        const int n_blocks = (SERIES_LENGTH - window_size + (block_size - 1)) / block_size;
        calculate_SADs<<<n_blocks, block_size>>>(d_values, d_SADs, d_filters, window_size, SERIES_LENGTH, query_idx);
        checkCudaErrors(cudaGetLastError());

    }

    // Wait for all kernels execution to finish
    checkCudaErrors(cudaDeviceSynchronize());
    // Copy results from GPU to CPU
    checkCudaErrors(cudaMemcpy(SADs.data(), d_SADs, filters.size() * SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

    // Free all resources
    cudaFree(d_filters);
    cudaFree(d_values);


    auto stop_benchmark = chrono::high_resolution_clock::now();
    auto duration_benchmark = chrono::duration_cast<chrono::milliseconds >(stop_benchmark - start_benchmark).count();
    
    cout << "Benchmark elapsed time: " << duration_benchmark << " ms" << endl;

    // Save data
    #pragma omp parallel for
    for(int query_idx = 0; query_idx < filters.size(); query_idx++){
        const vector<float> query = filters[query_idx];
        const int window_size = query.size();

        string file_name = "../output_data/SAD"+ to_string(query_idx) + "_filterlen" + to_string(query.size()) +".txt";
        ofstream output_file(file_name);
        if (output_file.is_open()) {
            for (int i = query_idx * SERIES_LENGTH; i < (query_idx + 1) * SERIES_LENGTH; i++){
                output_file << SADs[i] << "\n";
            }
            output_file.close();
            cout << "Saved successfully: " << file_name << endl;
        } else {
            cerr << "Failed to open: " << file_name << endl;
        }
    }

    // // Allocate space for the results
    // vector<float> means(SERIES_LENGTH, 0.0f);
    // vector<float> stds(SERIES_LENGTH, 0.0f);
    // map<int, vector<float>> znccs;

    // // Allocate GPU memory
    // float *d_values, *d_means, *d_stds, *d_znccs, *d_filter, *d_filt_mean, *d_filt_std;
    // checkCudaErrors(cudaMalloc((void**)&d_values, SERIES_LENGTH * sizeof(float)));
    // checkCudaErrors(cudaMalloc((void**)&d_means, SERIES_LENGTH * sizeof(float)));
    // checkCudaErrors(cudaMalloc((void**)&d_stds, SERIES_LENGTH * sizeof(float)));
    // checkCudaErrors(cudaMalloc((void**)&d_znccs, SERIES_LENGTH * sizeof(float)));
    // checkCudaErrors(cudaMalloc((void**)&d_filt_mean, 1 * sizeof(float)));
    // checkCudaErrors(cudaMalloc((void**)&d_filt_std, 1 * sizeof(float)));
    // // Copying data from CPU to GPU
    // checkCudaErrors(cudaMemcpy(d_values, values.data(), SERIES_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

    // for (int query_idx = 0; query_idx < filters.size(); query_idx++){
    //     const vector<float> query = filters[query_idx];
    //     const int window_size = query.size();
    //     float filt_mean, filt_std;

    //     // Calculate mean and std of filter
    //     float sum = 0.0f;
    //     for (int i = 0; i <= window_size; ++i) {
    //         sum += query[i];
    //     }
    //     filt_mean = sum / window_size;

    //     float variance_summation = 0.0f;
    //     for (int i = 0; i < window_size; ++i) {
    //         variance_summation += pow(query[i] - filt_mean, 2);
    //     }
    //     filt_std = sqrt(variance_summation / (window_size - 1));

    //     // Allocate and copy the filter values
    //     checkCudaErrors(cudaMalloc((void**)&d_filter, query.size() * sizeof(float)));
    //     checkCudaErrors(cudaMemcpy(d_filter, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
    //     checkCudaErrors(cudaMemcpy(d_filt_mean, &filt_mean, 1 * sizeof(float), cudaMemcpyHostToDevice));
    //     checkCudaErrors(cudaMemcpy(d_filt_std, &filt_std, 1 * sizeof(float), cudaMemcpyHostToDevice));

    //     calculate_means_windowed<<<n_blocks, block_size>>>(d_values, d_means, window_size, SERIES_LENGTH);
    //     checkCudaErrors(cudaGetLastError());
    //     // Wait for the kernel execution to finish
    //     checkCudaErrors(cudaDeviceSynchronize());

    //     calculate_stds_znccs_windowed<<<n_blocks, block_size>>>(d_values, d_means, d_stds, d_znccs, d_filter, d_filt_mean, d_filt_std, window_size, SERIES_LENGTH);
    //     checkCudaErrors(cudaGetLastError());
    //     // Wait for the kernel execution to finish
    //     checkCudaErrors(cudaDeviceSynchronize());


    //     // Copy results from GPU to CPU
    //     checkCudaErrors(cudaMemcpy(means.data(), d_means, SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
    //     checkCudaErrors(cudaMemcpy(stds.data(), d_stds, SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
    //     znccs[window_size] = vector<float>(SERIES_LENGTH, 0.0f);
    //     checkCudaErrors(cudaMemcpy(znccs[window_size].data(), d_znccs, SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));
        
    //     // Free space of the filter
    //     cudaFree(d_filter);
    // }
   

    // // Free all other resources
    // cudaFree(d_values);
    // cudaFree(d_means);
    // cudaFree(d_stds);
    // cudaFree(d_znccs);
    // cudaFree(d_filt_mean);
    // cudaFree(d_filt_std);

    // // Save data
    // #pragma omp parallel for
    // for(int query_idx = 0; query_idx < filters.size(); query_idx++){
    //     const vector<float> query = filters[query_idx];
    //     const int window_size = query.size();

    //     string file_name = "../output_data/zn_cross_correlation"+ to_string(query_idx) + "_filterlen" + to_string(query.size()) +".txt";
    //     ofstream output_file(file_name);
    //     if (output_file.is_open()) {
    //         for (float value : znccs[window_size]) {
    //             output_file << value << "\n";
    //         }
    //         output_file.close();
    //         cout << "Saved successfully: " << file_name << endl;
    //     } else {
    //         cerr << "Failed to open: " << file_name << endl;
    //     }
    // }


    return 0;
}