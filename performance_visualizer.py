import re
import os
import matplotlib.pyplot as plt
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