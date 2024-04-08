#include "utility.h"

int main() {
    // Check if output folder exists and if not it creates the folder
    string output_folder = "../output_data";
    if (!fs::exists(output_folder)) {
        // Crea la cartella
        fs::create_directory(output_folder);
        std::cout << "Folder " << output_folder << " created\n";
    }

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
    vector<float> tmp = create_filter_trend(FILTER_LENGTH, true);
    for(int i = 0; i < N_FILTERS; i++){
        temp_filters.insert(temp_filters.end(), tmp.begin(), tmp.end());
    }
    vector<float> filters = temp_filters;

    // Execute benchmark with different number of threads
    for(int n_threads = 64; n_threads <= 64; n_threads += 8){
        omp_set_num_threads(n_threads);
        
        // Execute multiple test with same number of threads   
        for(int n_test = 0; n_test < N_TEST; n_test++){
            
            cout << "Executing benchmark..." << endl;

            if(CPU_flt_type == "SAD"){
                auto start_benchmark = chrono::high_resolution_clock::now();

                // Create space for the results
                vector<float> SADs(N_FILTERS * SERIES_LENGTH, 0.0f);

                #pragma omp parallel for
                for(int central_idx = FILTER_LENGTH / 2; central_idx < SERIES_LENGTH - FILTER_LENGTH; central_idx++){
                    int start_window_idx = central_idx - FILTER_LENGTH / 2;
                    // Create space for temporary values
                    vector<float> SAD(N_FILTERS, 0.0f);

                    //Compute (multiple) SAD for this idx applying (multiple) filter 
                    for (int i = 0; i <= FILTER_LENGTH; ++i) {
                        const float current_value = values[start_window_idx + i];
                        for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                            // Save result in temporary memory
                            SAD[filter_idx] += abs(current_value - filters[i + filter_idx * FILTER_LENGTH]);
                        }
                    }
                    // Copy results in final memory
                    for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                        SADs[central_idx + filter_idx * SERIES_LENGTH] = SAD[filter_idx] / FILTER_LENGTH;
                    }
                }

                auto stop_benchmark = chrono::high_resolution_clock::now();
                auto duration_benchmark = chrono::duration_cast<chrono::milliseconds >(stop_benchmark - start_benchmark).count();
                
                cout << "Benchmark elapsed time: " << duration_benchmark << " ms" << endl;

                // Save data
                if (save_data){
                    #pragma omp parallel for
                    for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                        string file_name = "../output_data/SAD"+ to_string(filter_idx) + "_filterlen" + to_string(FILTER_LENGTH) + ".txt";
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
                }

                // Write benchmark result to a file
                string out = "../output_data/benchmark_results.txt";
                std::ofstream f(out, std::ios::app);
                if(f.is_open()) {
                    std::time_t currentTime = std::time(nullptr);

                    // Format data and time in a string
                    char buffer[80];
                    std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));

                    f << buffer << "   Series length: " << SERIES_LENGTH << "   Filter length: " << FILTER_LENGTH << "   Filter type: "<< CPU_flt_type << "   Elapsed: " << duration_benchmark << " ms"  << "   N threads: " << n_threads << endl;
                    
                    std::cout << "Benchmark saved successfully" << std::endl;
                }
                else {
                    std::cerr << "Unable to open the file" << std::endl;
                    exit(-1);
                }

            }
            
            if(CPU_flt_type == "ZMNCC"){

                // Allocate CPU memory for the results
                vector<float> means(SERIES_LENGTH, 0.0f);
                vector<float> stds(SERIES_LENGTH, 0.0f);
                vector<float> zmnccs(N_FILTERS * SERIES_LENGTH, 0.0f);
                vector<float> filt_means(N_FILTERS, 0.0f);
                vector<float> filt_stds(N_FILTERS, 0.0f);

                // Calculate mean and std of filters
                #pragma omp parallel for
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

                auto start_benchmark = chrono::high_resolution_clock::now();

                // Parallel execution
                calculate_means_windowed(values, means, FILTER_LENGTH, SERIES_LENGTH);
                
                // Parallel execution
                calculate_stds_zmnccs_windowed(values, means, stds, zmnccs, filters, filt_means, filt_stds, FILTER_LENGTH, SERIES_LENGTH);

                auto stop_benchmark = chrono::high_resolution_clock::now();
                auto duration_benchmark = chrono::duration_cast<chrono::milliseconds >(stop_benchmark - start_benchmark).count();     
                
                cout << "Benchmark elapsed time: " << duration_benchmark << " ms" << endl;
                
                // Save data
                if (save_data){
                    #pragma omp parallel for
                    for(int filter_idx = 0; filter_idx < N_FILTERS; filter_idx++){
                        string file_name = "../output_data/zmncc"+ to_string(filter_idx) + "_filterlen" + to_string(FILTER_LENGTH) +".txt";
                        ofstream output_file(file_name);
                        if (output_file.is_open()) {
                            for (int i = 0; i < SERIES_LENGTH; i++){
                                output_file << zmnccs[filter_idx * SERIES_LENGTH + i] << "\n";
                            }
                            output_file.close();
                            cout << "Saved successfully: " << file_name << endl;
                        } else {
                            cerr << "Failed to open: " << file_name << endl;
                        }
                    }
                }

                // Write benchmark result to a file
                string out = "../output_data/benchmark_results.txt";
                std::ofstream f(out, std::ios::app);
                if(f.is_open()) {
                    std::time_t currentTime = std::time(nullptr);

                    // Format data and time in a string
                    char buffer[80];
                    std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", std::localtime(&currentTime));

                    f << buffer << "   Series length: " << SERIES_LENGTH << "   Filter length: " << FILTER_LENGTH << "   Filter type: "<< CPU_flt_type << "   Elapsed: " << duration_benchmark << " ms" << "   N threads: " << n_threads  << endl;
                    
                    std::cout << "Benchmark saved successfully" << std::endl;
                }
                else {
                    std::cerr << "Unable to open the file" << std::endl;
                    exit(-1);
                }
            }    
        }
    }

    return 0;
}
