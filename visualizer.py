import re
import os
import matplotlib.pyplot as plt
import time
import sqlite3
from datetime import datetime
plt.rcParams.update({'font.size': 16})

def extract_elaps_avg(data: list[tuple]):
    # Initialize an empty dictionary to store the average elapsed time for each number of threads
    average_elapsed_times = {}

    # Iterate through the result set
    for row in data:
        n_threads = row[0]
        elapsed_time = row[1]
        
        # If the number of threads is already in the dictionary append the elapsed time
        if n_threads in average_elapsed_times:
            average_elapsed_times[n_threads].append(elapsed_time)
        # If the number of threads is not in the dictionary, initialize a list with the elapsed time
        else:
            average_elapsed_times[n_threads] = [elapsed_time]

    # Calculate the average elapsed time for each number of threads
    for n_threads, elapsed_times in average_elapsed_times.items():
        average_elapsed_times[n_threads] = sum(elapsed_times) / len(elapsed_times)
    
    # Sort the dictionary based on the value of the keys
    sorted_keys = sorted(average_elapsed_times.keys())
    sorted_dict = {key: average_elapsed_times[key] for key in sorted_keys}

    return sorted_dict

def custom_sort(string):
    # Extract the number from the string using regular expressions
    number = re.search(r'\d+', string).group()
    # Convert the number to an integer and return it for sorting
    return int(number)


os.chdir("./")
output_folder = "output_data"
output_visualizer_folder = "output_visualizer"
database_file = "benchmark_data.db"
benchmark_file = "benchmark_results.txt"
path_to_db = os.path.join(output_folder, database_file)
path_to_benchmark = os.path.join(output_folder, benchmark_file)

pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+Series length: (\d+)\s+Filter length: (\d+)\s+Filter type: (\w+)\s+Elapsed: (\d+) ms(?:\s+N threads: (\d+))?'
str_datetime_format = "%Y-%m-%d %H:%M:%S"

# Check if the folder exist already
if not os.path.exists(output_visualizer_folder):
    # Create the folder if it doesn't exist
    os.makedirs(output_visualizer_folder)
    print(f"Folder created successfully: {output_visualizer_folder}")

# Connect to the database (if it doesn't exist, it will be created automatically)
conn = sqlite3.connect(path_to_db)

# Create a cursor to execute queries
c = conn.cursor()

# Create the table if it doesn't exist already
c.execute('''CREATE TABLE IF NOT EXISTS benchmarks
             (timestamp DATETIME, series_length INTEGER, filter_length INTEGER, filter_type TEXT, elapsed INTEGER, n_threads INTEGER)''')

# Check the timestamp of the last entry in the database
c.execute('SELECT MAX(timestamp) FROM benchmarks')
last_date_time = c.fetchone()[0]
if last_date_time is not None:
    last_date_time = datetime.strptime(last_date_time, str_datetime_format)

# Read data from the file and insert only new data
with open(path_to_benchmark, 'r') as file:
    for line in file:
        match = re.match(pattern, line)
        date_time = match.group(1)
        series_length = int(match.group(2))
        filter_length = int(match.group(3))
        filter_type = match.group(4)
        elapsed_time = int(match.group(5))
        num_threads = int(match.group(6)) if match.group(6) else -1
        timestamp = datetime.strptime(date_time, str_datetime_format)
        data = (timestamp, series_length, filter_length, filter_type, elapsed_time, num_threads)

        # Insert data into the database if it's newer than the last entry
        if last_date_time is None or timestamp > last_date_time:
            c.execute('INSERT INTO benchmarks VALUES (?,?,?,?,?,?)', data)

# Save changes
conn.commit()               

# Graph 1
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_64_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "SAD", 64)).fetchall()
query_32_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "SAD", 32)).fetchall()
query_16_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "SAD", 16)).fetchall()
query_CUDA_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "SAD_CUDA", -1)).fetchall()
dict_query_64_data = extract_elaps_avg(data=query_64_data)
dict_query_32_data = extract_elaps_avg(data=query_32_data)
dict_query_16_data = extract_elaps_avg(data=query_16_data)
dict_query_CUDA_data = extract_elaps_avg(data=query_CUDA_data)

ax.plot(dict_query_64_data.keys(), dict_query_64_data.values(), label="OpenMP (64 threads)", marker=".", markersize=12)
ax.plot(dict_query_32_data.keys(), dict_query_32_data.values(), label="OpenMP (32 threads)", marker=".", markersize=12)
ax.plot(dict_query_16_data.keys(), dict_query_16_data.values(), label="OpenMP (16 threads)", marker=".", markersize=12)
ax.plot(dict_query_CUDA_data.keys(), dict_query_CUDA_data.values(), label="CUDA", marker=".", markersize=12)
ax.set_ylabel('ms')
ax.set_xlabel('Series length')
ax.set_title("SAD performance")
ax.legend()
ax.grid(True, which="both")
plt.savefig(os.path.join(output_visualizer_folder, "SAD_length"))

# Graph 2
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_64_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (20752600, 12032, "SAD", 64)).fetchall()
query_32_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (20752600, 12032, "SAD", 32)).fetchall()
query_16_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (20752600, 12032, "SAD", 16)).fetchall()
query_CUDA_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (20752600, 12032, "SAD_CUDA", -1)).fetchall()
dict_query_64_data = extract_elaps_avg(data=query_64_data)
dict_query_32_data = extract_elaps_avg(data=query_32_data)
dict_query_16_data = extract_elaps_avg(data=query_16_data)
dict_query_CUDA_data = extract_elaps_avg(data=query_CUDA_data)

ax.bar(0, dict_query_64_data.values(), label="OpenMP (64 threads)")
ax.bar(1, dict_query_32_data.values(), label="OpenMP (32 threads)")
ax.bar(2, dict_query_16_data.values(), label="OpenMP (16 threads)")
ax.bar(3, dict_query_CUDA_data.values(), label="CUDA")
ax.set_ylabel('ms')
ax.set_xlabel('Lower is better')
ax.set_title("SAD performance")
ax.set_xticks([])
ax.legend()
ax.grid(axis="y")
plt.savefig(os.path.join(output_visualizer_folder, "SADvs"))

# Graph 3
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_64_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "ZMNCC", 64)).fetchall()
query_32_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "ZMNCC", 32)).fetchall()
query_16_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "ZMNCC", 16)).fetchall()
query_CUDA_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "ZMNCC_CUDA_NO", -1)).fetchall()
query_CUDA_SH_data = c.execute("SELECT series_length, elapsed FROM benchmarks WHERE filter_length = ? AND filter_type = ? AND n_threads = ?", (12032, "ZMNCC_CUDA_SHARED", -1)).fetchall()
dict_query_64_data = extract_elaps_avg(data=query_64_data)
dict_query_32_data = extract_elaps_avg(data=query_32_data)
dict_query_16_data = extract_elaps_avg(data=query_16_data)
dict_query_CUDA_data = extract_elaps_avg(data=query_CUDA_data)
dict_query_CUDA_SH_data = extract_elaps_avg(data=query_CUDA_SH_data)

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

query_64_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (6225780, 12032, "ZMNCC", 64)).fetchall()
query_32_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (6225780, 12032, "ZMNCC", 32)).fetchall()
query_16_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (6225780, 12032, "ZMNCC", 16)).fetchall()
query_CUDA_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (6225780, 12032, "ZMNCC_CUDA_NO", -1)).fetchall()
query_CUDA_SH_data = c.execute("SELECT filter_type, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ? AND n_threads = ?", (6225780, 12032, "ZMNCC_CUDA_SHARED", -1)).fetchall()
dict_query_64_data = extract_elaps_avg(data=query_64_data)
dict_query_32_data = extract_elaps_avg(data=query_32_data)
dict_query_16_data = extract_elaps_avg(data=query_16_data)
dict_query_CUDA_data = extract_elaps_avg(data=query_CUDA_data)
dict_query_CUDA_SH_data = extract_elaps_avg(data=query_CUDA_SH_data)

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

# Graph 5
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_data = c.execute("SELECT n_threads, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ?", (2075260, 12032, "ZMNCC")).fetchall()
dict_query_data = extract_elaps_avg(data=query_data)

ax.plot(dict_query_data.keys(), dict_query_data.values(), label="OpenMP", marker=".", markersize=12)
ax.set_ylabel('ms')
ax.set_xlabel('# of threads')
ax.set_title("ZMNCC performance")
ax.legend()
ax.grid(True, which="both")
plt.savefig(os.path.join(output_visualizer_folder, "ZMNCC_nthreads"))

# Graph 6
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

query_data = c.execute("SELECT n_threads, elapsed FROM benchmarks WHERE series_length = ? AND filter_length = ? AND filter_type = ?", (20752600, 12032, "SAD")).fetchall()
dict_query_data = extract_elaps_avg(data=query_data)

ax.plot(dict_query_data.keys(), dict_query_data.values(), label="OpenMP", marker=".", markersize=12)
ax.set_ylabel('ms')
ax.set_xlabel('# of threads')
ax.set_title("SAD performance")
ax.legend()
ax.grid(True, which="both")
plt.savefig(os.path.join(output_visualizer_folder, "SAD_nthreads"))

# Close the connection to the database
conn.close()     

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