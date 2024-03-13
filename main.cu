#include "cuda_utility.h"

int main() {
    string flt_type = "ZMNCC_CUDA"; // "SAD_CUDA" or "ZMNCC_CUDA"
    string shared = "SHARED"; // "SHARED": use shared memory

    // Open the file
    ifstream file("../input_data/input.txt");
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

    // Create a bank of filters
    vector<float> temp_filters;

    vector<float> tmp = create_filter(FILTER_LENGTH);

    for(int i = 0; i < N_FILTERS; i++){
        temp_filters.insert(temp_filters.end(), tmp.begin(), tmp.end());
    }
    const vector<float> filters = temp_filters;

    for(int n_test = 0; n_test < 3; n_test++){

        cout << "Executing benchmark..." << endl;
    
        if(flt_type == "SAD_CUDA"){
            auto start_benchmark = chrono::high_resolution_clock::now();

            const int block_size = 256;

            // Allocate CPU space for the result
            vector<float> SADs(N_FILTERS * SERIES_LENGTH, 0.0f);

            // Allocate GPU space for data values, results and filters
            float *d_values, *d_SADs, *d_filters;
            checkCudaErrors(cudaMalloc((void**)&d_values, SERIES_LENGTH * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&d_SADs, N_FILTERS * SERIES_LENGTH * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&d_filters, filters.size() * sizeof(float)));

            // Copying data values and filters from CPU to GPU
            checkCudaErrors(cudaMemcpy(d_values, values.data(), SERIES_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_filters , filters.data(), filters.size() * sizeof(float), cudaMemcpyHostToDevice));

            const int n_blocks = (SERIES_LENGTH - FILTER_LENGTH + (block_size - 1)) / block_size;
            calculate_SADs<<<n_blocks, block_size>>>(d_values, d_SADs, d_filters, FILTER_LENGTH, SERIES_LENGTH);
            checkCudaErrors(cudaGetLastError());

            // Wait for all kernels execution to finish
            checkCudaErrors(cudaDeviceSynchronize());
            // Copy results from GPU to CPU
            checkCudaErrors(cudaMemcpy(SADs.data(), d_SADs, N_FILTERS * SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

            // Free all resources
            cudaFree(d_filters);
            cudaFree(d_values);
            cudaFree(d_SADs);

            auto stop_benchmark = chrono::high_resolution_clock::now();
            auto duration_benchmark = chrono::duration_cast<chrono::milliseconds >(stop_benchmark - start_benchmark).count();
            
            cout << "Benchmark elapsed time: " << duration_benchmark << " ms" << endl;

            // Save data
            for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                string file_name = "../output_data/SAD"+ to_string(filter_idx) + "_filterlen" + to_string(FILTER_LENGTH) +".txt";
                ofstream output_file(file_name);
                if (output_file.is_open()) {
                    for (int i = 0; i < SERIES_LENGTH; i++){
                        output_file << SADs[filter_idx * SERIES_LENGTH + i] << "\n";
                    }
                    output_file.close();
                    cout << "Saved successfully: " << file_name << endl;
                } else {
                    cerr << "Failed to open: " << file_name << endl;
                }
            }

            // Write benchmark result to a file
            string out = "../output_data/benchmark_results_CUDA.txt";
            std::ofstream f(out, std::ios::app);
            if(f.is_open()) {
                std::time_t currentTime = std::time(nullptr);

                // Format data and time in a string
                char buffer[80];
                std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));

                f << "[" << buffer << "] Series length: " << SERIES_LENGTH << "   Filter length: " << FILTER_LENGTH << "   Filter type: "<< flt_type << "   Elapsed: " << duration_benchmark << " ms" << endl;
                
                std::cout << "Data saved successfully" << std::endl;
            }
            else {
                std::cerr << "Unable to open the file" << std::endl;
                exit(-1);
            }


        }
        else if(flt_type == "ZMNCC_CUDA"){

            // Allocate CPU memory for the results
            vector<float> means(SERIES_LENGTH, 0.0f);
            vector<float> stds(SERIES_LENGTH, 0.0f);
            vector<float> zmnccs(N_FILTERS * SERIES_LENGTH, 0.0f);
            vector<float> filt_means(N_FILTERS, 0.0f);
            vector<float> filt_stds(N_FILTERS, 0.0f);

            // Allocate GPU memory
            float *d_values, *d_means, *d_stds, *d_zmnccs, *d_filters, *d_filt_means, *d_filt_stds;
            checkCudaErrors(cudaMalloc((void**)&d_values, SERIES_LENGTH * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&d_means, SERIES_LENGTH * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&d_stds, SERIES_LENGTH * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&d_zmnccs, N_FILTERS * SERIES_LENGTH * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&d_filters, N_FILTERS * FILTER_LENGTH * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&d_filt_means, N_FILTERS * sizeof(float)));
            checkCudaErrors(cudaMalloc((void**)&d_filt_stds, N_FILTERS * sizeof(float)));
            
            // Copying data from CPU to GPU
            checkCudaErrors(cudaMemcpy(d_values, values.data(), SERIES_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_filters, filters.data(), N_FILTERS * FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice));

            // Calculate mean and std of filters
            for (int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                const int window_size = FILTER_LENGTH;
                float filt_mean, filt_std;

                float sum = 0.0f;
                for (int i = 0; i <= window_size; ++i) {
                    sum += filters[i + filter_idx * FILTER_LENGTH];
                }
                filt_mean = sum / window_size;

                float variance_summation = 0.0f;
                for (int i = 0; i < window_size; ++i) {
                    variance_summation += pow(filters[i + filter_idx * FILTER_LENGTH] - filt_mean, 2);
                }
                filt_std = sqrt(variance_summation / (window_size - 1));
                
                // Save the mean and std of filter
                filt_means[filter_idx] = filt_mean;
                filt_stds[filter_idx] = filt_std;
            }
            cout << "Finished computing means and stds of filters" << endl;

            // Copying data from CPU to GPU
            checkCudaErrors(cudaMemcpy(d_filt_means, filt_means.data(), N_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_filt_stds, filt_stds.data(), N_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
            
            auto start_benchmark = chrono::high_resolution_clock::now();
            
            const int block_size = 256;
            const int n_blocks = (SERIES_LENGTH - FILTER_LENGTH + (block_size - 1)) / block_size;
            const int values_to_load_per_thread = (FILTER_LENGTH + block_size) / block_size;
            const int threads_for_loading = (n_blocks * block_size) / values_to_load_per_thread;

            // Parallel execution 
            calculate_means_windowed<<<n_blocks, block_size>>>(d_values, d_means, FILTER_LENGTH, SERIES_LENGTH);
            checkCudaErrors(cudaGetLastError());
            // Wait for the kernel execution to finish to compute means
            checkCudaErrors(cudaDeviceSynchronize());

            if(shared == "SHARED"){
                calculate_stds_zmnccs_windowed_shared<<<n_blocks, block_size>>>(d_values, d_means, d_stds, d_zmnccs, d_filters, d_filt_means, d_filt_stds, FILTER_LENGTH, SERIES_LENGTH, values_to_load_per_thread, threads_for_loading);
            }
            else{
                calculate_stds_zmnccs_windowed<<<n_blocks, block_size>>>(d_values, d_means, d_stds, d_zmnccs, d_filters, d_filt_means, d_filt_stds, FILTER_LENGTH, SERIES_LENGTH);
            }
            checkCudaErrors(cudaGetLastError());
            // Wait for the kernel execution to finish compute stds and zmnccs
            checkCudaErrors(cudaDeviceSynchronize());

            // Copy results from GPU to CPU
            checkCudaErrors(cudaMemcpy(zmnccs.data(), d_zmnccs, N_FILTERS * SERIES_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));  

            auto stop_benchmark = chrono::high_resolution_clock::now();
            auto duration_benchmark = chrono::duration_cast<chrono::milliseconds >(stop_benchmark - start_benchmark).count();
            
            cout << "Benchmark elapsed time: " << duration_benchmark << " ms" << endl;

            // Free all resources
            cudaFree(d_values);
            cudaFree(d_means);
            cudaFree(d_stds);
            cudaFree(d_zmnccs);
            cudaFree(d_filters);
            cudaFree(d_filt_means);
            cudaFree(d_filt_stds);

            // Save data
            for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                string file_name = "../output_data/zmncc"+ to_string(filter_idx) + "_filterlen" + to_string(FILTER_LENGTH) +".txt";
                ofstream output_file(file_name);
                if (output_file.is_open()) {
                    for (int i = 0; i < SERIES_LENGTH; i++) {
                        output_file << zmnccs[filter_idx * SERIES_LENGTH + i] << "\n";
                    }
                    output_file.close();
                    cout << "Saved successfully: " << file_name << endl;
                } else {
                    cerr << "Failed to open: " << file_name << endl;
                }
            }

            // Write benchmark result to a file
            string out = "../output_data/benchmark_results_CUDA.txt";
            std::ofstream f(out, std::ios::app);
            if(f.is_open()) {
                std::time_t currentTime = std::time(nullptr);

                // Format data and time in a string
                char buffer[80];
                std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));

                f << "[" << buffer << "] Series length: " << SERIES_LENGTH << "   Filter length: " << FILTER_LENGTH << "   Filter type: "<< flt_type + "_" + shared << "   Elapsed: " << duration_benchmark << " ms" << endl;
                
                std::cout << "Data saved successfully" << std::endl;
            }
            else {
                std::cerr << "Unable to open the file" << std::endl;
                exit(-1);
            }
        }
    }

    return 0;
}