import re
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
plt.rcParams.update({'font.size': 16})

N_TEST = 3

def extract_data_avg(data: list[tuple], n_test=N_TEST):
    dict_query_64_data = {}
    for data in data:
        if data[0] not in dict_query_64_data.keys():
            dict_query_64_data[data[0]] = data[3]
        else:
            dict_query_64_data[data[0]] += data[3]
    for key in dict_query_64_data:
        dict_query_64_data[key] /= n_test
    return dict_query_64_data

def custom_sort(string):
    # Extract the number from the string using regular expressions
    number = re.search(r'\d+', string).group()
    # Convert the number to an integer and return it for sorting
    return int(number)

os.chdir("./")

# Check if the folder exist already
output_visualizer_folder = "output_visualizer"
if not os.path.exists(output_visualizer_folder):
    # Create the folder if it doesn't exist
    os.makedirs(output_visualizer_folder)
    print(f"Folder created successfully: {output_visualizer_folder}")

# Read all data
output_folder = "output_data"

files = os.listdir(output_folder)

benchmark_data = []
benchmark_CUDA_data = []
# Extract data
for file_name in files:
    if "benchmark" in file_name:
        # Read file
        with open(os.path.join(output_folder, file_name), "r") as file:
            for line in file:
                if file_name == "benchmark_results_CUDA.txt":

                    # Benchmark CUDA data idxs:  0                      1                                    2                                   3          4: N threads = 0
                    pattern = r"Series length: (\d+)   Filter length: (\d+)   Filter type: \b(ZMNCC_CUDA_NO|SAD_CUDA|ZMNCC_CUDA_SHARED)   Elapsed: (\d+) ms"
                    match = re.search(pattern=pattern, string=line)
                    if match:
                        benchmark_CUDA_data.append(( int(match.group(1)), int(match.group(2)), match.group(3), int(match.group(4)), 0 ))

                elif file_name == "benchmark_results.txt":
                    # Benchmark data indexes:     0                     1                    2                3                     4
                    pattern = r"Series length: (\d+)   Filter length: (\d+)   Filter type: (\w+)   Elapsed: (\d+) ms   N threads: (\d+)"
                    match = re.search(pattern=pattern, string=line)
                    if match:
                        benchmark_data.append(( int(match.group(1)), int(match.group(2)), match.group(3), int(match.group(4)), int(match.group(5)) ))                       

# Graph 1
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_64_data = [data for data in benchmark_data if data[1] == 12032 and data[2] == "SAD" and data[4] == 64]
query_CUDA_data = [data for data in benchmark_CUDA_data if data[1] == 12032 and data[2] == "SAD_CUDA"]
dict_query_64_data = extract_data_avg(data=query_64_data)
dict_query_CUDA_data = extract_data_avg(data=query_CUDA_data)

ax.plot(dict_query_64_data.keys(), dict_query_64_data.values(), label="OpenMP (64 threads)", marker=".", markersize=12)
ax.plot(dict_query_CUDA_data.keys(), dict_query_CUDA_data.values(), label="CUDA", marker=".", markersize=12)
ax.set_ylabel('ms')
ax.set_xlabel('Series length')
ax.set_title("SAD performance")
ax.legend()
ax.grid(True, which="both")
plt.savefig(os.path.join(output_visualizer_folder, "SAD_length"))

# Graph 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_64_data = [data for data in benchmark_data if data[0] == 20752600 and data[1] == 12032 and data[2] == "SAD" and data[4] == 64]
query_CUDA_data = [data for data in benchmark_CUDA_data if data[0] == 20752600 and data[1] == 12032 and data[2] == "SAD_CUDA"]
dict_query_64_data = extract_data_avg(data=query_64_data)
dict_query_CUDA_data = extract_data_avg(data=query_CUDA_data)

ax.bar(0, dict_query_64_data.values(), label="OpenMP (64 threads)")
ax.bar(1, dict_query_CUDA_data.values(), label="CUDA")
ax.set_ylabel('ms')
ax.set_xlabel('Lower is better')
ax.set_title("SAD performance")
ax.set_xticks([])
ax.legend()
ax.grid(axis="y")
plt.savefig(os.path.join(output_visualizer_folder, "SADvs"))

# Graph 3
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_64_data = [data for data in benchmark_data if data[1] == 12032 and data[2] == "ZMNCC" and data[4] == 64]
query_32_data = [data for data in benchmark_data if data[1] == 12032 and data[2] == "ZMNCC" and data[4] == 32]
query_16_data = [data for data in benchmark_data if data[1] == 12032 and data[2] == "ZMNCC" and data[4] == 16]
query_CUDA_data = [data for data in benchmark_CUDA_data if data[1] == 12032 and data[2] == "ZMNCC_CUDA_NO"]
query_CUDA_SH_data = [data for data in benchmark_CUDA_data if data[1] == 12032 and data[2] == "ZMNCC_CUDA_SHARED"]
dict_query_64_data = extract_data_avg(data=query_64_data)
dict_query_32_data = extract_data_avg(data=query_32_data)
dict_query_16_data = extract_data_avg(data=query_16_data)
dict_query_CUDA_data = extract_data_avg(data=query_CUDA_data)
dict_query_CUDA_SH_data = extract_data_avg(data=query_CUDA_SH_data)

ax.plot(dict_query_64_data.keys(), dict_query_64_data.values(), label="OpenMP (64 threads)", marker=".", markersize=12)
ax.plot(dict_query_32_data.keys(), dict_query_32_data.values(), label="OpenMP (32 threads)", marker=".", markersize=12)
ax.plot(dict_query_16_data.keys(), dict_query_16_data.values(), label="OpenMP (16 threads)", marker=".", markersize=12)
ax.plot(dict_query_CUDA_data.keys(), dict_query_CUDA_data.values(), label="CUDA", marker=".", markersize=12)
ax.plot(dict_query_CUDA_SH_data.keys(), dict_query_CUDA_SH_data.values(), label="CUDA shared mem", marker=".", markersize=12)
ax.set_ylabel('ms')
ax.set_xlabel('Series length')
ax.set_title("ZMNCC performance")
ax.legend()
ax.grid(True, which="both")
plt.savefig(os.path.join(output_visualizer_folder, "ZMNCC_length"))

# Graph 4
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_64_data = [data for data in benchmark_data if data[0] == 6225780 and data[1] == 12032 and data[2] == "ZMNCC" and data[4] == 64]
query_32_data = [data for data in benchmark_data if data[1] == 12032 and data[2] == "ZMNCC" and data[4] == 32]
query_16_data = [data for data in benchmark_data if data[1] == 12032 and data[2] == "ZMNCC" and data[4] == 16]
query_CUDA_data = [data for data in benchmark_CUDA_data if data[0] == 6225780 and data[1] == 12032 and data[2] == "ZMNCC_CUDA_NO"]
query_CUDA_SH_data = [data for data in benchmark_CUDA_data if data[0] == 6225780 and data[1] == 12032 and data[2] == "ZMNCC_CUDA_SHARED"]
dict_query_64_data = extract_data_avg(data=query_64_data)
dict_query_32_data = extract_data_avg(data=query_32_data)
dict_query_16_data = extract_data_avg(data=query_16_data)
dict_query_CUDA_data = extract_data_avg(data=query_CUDA_data)
dict_query_CUDA_SH_data = extract_data_avg(data=query_CUDA_SH_data)

ax.bar(0, dict_query_64_data.values(), label="OpenMP (64 threads)")
ax.bar(1, dict_query_32_data.values(), label="OpenMP (32 threads)")
ax.bar(2, dict_query_16_data.values(), label="OpenMP (16 threads)")
ax.bar(3, dict_query_CUDA_data.values(), label="CUDA")
ax.bar(4, dict_query_CUDA_SH_data.values(), label="CUDA shared mem")
ax.set_ylabel('ms')
ax.set_xlabel('Lower is better')
ax.set_title("ZMNCC performance")
ax.set_xticks([])
ax.legend()
ax.grid(axis="y")
plt.savefig(os.path.join(output_visualizer_folder, "ZMNCCvs"))

# date_data : list[datetime] = []
# power_data : list[float] = [] 

# format = "%d/%m/%Y %H:%M:%S"

# with open("input_data/household_power_consumption_short.txt", "r") as file:
#     # Skip the header
#     next(file)
    
#     # Read each line in the file
#     for line in file:
#         # Split the line into columns
#         columns = line.strip().split(';')
        
#         # Extract "Date Time"  and "Global_active_power"
#         date_time = columns[0] + ' ' + columns[1]  # Combine Date and Time

#         converted_date_time = datetime.strptime(date_time, format)
#         # Manage missing power data (symbol "?"), put a 0
#         if columns[2] != "?":
#             power = float(columns[2])  # Convert Global_active_power to float
#         else:
#             power = 0.0
        
#         # Append the extracted data to the respective lists
#         date_data.append(converted_date_time)
#         power_data.append(power)

# # Sorting the list using custom_sort function as key
# list_of_strings = os.listdir(output_folder)
# files = sorted(list_of_strings, key=custom_sort)

# zmnccs_data = []

# for zmnccs_file in files:
#     temp = []
#     pattern = r"zmnccs\d+_filterlen(\d+)"
#     filter_length = int(re.search(pattern=pattern, string=zmnccs_file).group(1))
#     offset = int(filter_length / 2)
#     # print(f"File: {zmnccs_file}, filterlen: {filter_length}, offset: {int(filter_length / 2)}")

#     with open(os.path.join(output_folder, zmnccs_file), 'r') as file:
#         # Add zero padding at the beginning
#         for i in range(offset):
#             temp.append((i, 0))
#         # Get data from file
#         for i, line in enumerate(file, 1):
#             value = float(line.strip())
#             temp.append((i + offset, value))
#             last_iter = i
#         # Add zero padding at the end
#         for i in range(offset - 1):
#             temp.append((last_iter + i, 0))

#     zmnccs_data.append(temp)

# print("Start plotting")

# # Plotting
# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(70, 20))

# start_time = time.time()
# axes[0].plot(date_data[::720], power_data[::720], label="Household Active Power")
# axes[0].set_ylabel('kW')
# axes[0].set_title("Household Active Power")
# axes[0].set_xticks(date_data[::133920])
# axes[0].set_xticks(date_data[::44640], minor=True)
# axes[0].grid(True, which="both")


# axes[1].plot(date_data[::720], [data[1] for data in zmnccs_data[0][::720]], label="...")
# axes[1].set_ylabel('Uptrend 1 week')
# axes[1].set_xticks(date_data[::133920])
# axes[1].set_xticks(date_data[::44640], minor=True)
# axes[1].grid(True, which="both")
# axes[1].axhline(y=0, color="black", linestyle=(0, (5, 10)), alpha=0.5)

# axes[2].plot(date_data[::720], [data[1] for data in zmnccs_data[1][::720]], label="...")
# axes[2].set_ylabel('Uptrend 4 weeks')
# axes[2].set_xticks(date_data[::133920])
# axes[2].set_xticks(date_data[::44640], minor=True)
# axes[2].grid(True, which="both")
# axes[2].axhline(y=0, color="black", linestyle=(0, (5, 10)), alpha=0.5)

# axes[3].plot(date_data[::720], [data[1] for data in zmnccs_data[2][::720]], label="...")
# axes[3].set_ylabel('Uptrend 8 weeks')
# axes[3].set_xticks(date_data[::133920])
# axes[3].set_xticks(date_data[::44640], minor=True)
# axes[3].grid(True, which="both")
# axes[3].axhline(y=0, color="black", linestyle=(0, (5, 10)), alpha=0.5)

# end_time = time.time()
# elapsed_time = end_time - start_time

# print(f"Time elapsed to plot: {elapsed_time} s")

# # plt.tight_layout()
# plt.savefig('./output_visualizer/cross-corr2')