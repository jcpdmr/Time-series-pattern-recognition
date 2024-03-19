import re
import os
import matplotlib.pyplot as plt
import time
import statistics
import numpy as np
from datetime import datetime
plt.rcParams.update({'font.size': 10})

output_folder = "output_data"
FILTER_LENGTH = 12032
SUBSAMPLING = FILTER_LENGTH // 3
AVG_WINDOW = 3
SKIP = AVG_WINDOW  // 2

date_data : list[datetime] = []
power_data : list[float] = [] 
format = "%d/%m/%Y %H:%M:%S"
print("Retrieving dataset...")
with open("input_data/household_power_consumption.txt", "r") as file:
    # Skip the header
    next(file)  
    # Read each line in the file
    for line in file:
        # Split the line into columns
        columns = line.strip().split(';')   
        # Extract "Date Time"  and "Global_active_power"
        date_time = columns[0] + ' ' + columns[1]  # Combine Date and Time
        converted_date_time = datetime.strptime(date_time, format)
        # Manage missing power data (symbol "?"), put a 0
        if columns[2] != "?":
            power = float(columns[2])  # Convert Global_active_power to float
        else:
            power = 0.0     
        # Append the extracted data to the respective lists
        date_data.append(converted_date_time)
        power_data.append(power)

files = os.listdir(output_folder)

zmncc_results = {}
sad_results = {}
print("Retrieving output data...")
for file in files:
    filt_res_val = []
    if "zmncc" in file:
        #Get configuration info
        pattern = r"zmncc(\d+)_filterlen(\d+)"
        match = re.search(pattern=pattern, string=file)
        filter_id = int(match.group(1))
        filter_length = int(match.group(2))
        # Read values in the file
        with open(os.path.join(output_folder, file), 'r') as file:
            for line in file:
                value = float(line.strip())
                filt_res_val.append(value)
        # Save in the dictionary
        zmncc_results[filter_id] = filt_res_val
    
    elif "SAD" in file:
        #Get configuration info
        pattern = r"SAD(\d+)_filterlen(\d+)"
        match = re.search(pattern=pattern, string=file)
        filter_id = int(match.group(1))
        filter_length = int(match.group(2))
        # Read values in the file
        with open(os.path.join(output_folder, file), 'r') as file:
            for line in file:
                value = float(line.strip())
                filt_res_val.append(value)
        # Save in the dictionary
        sad_results[filter_id] = filt_res_val
    
    else:
        # Found a file that it's not a SAD or ZMNCC result
        pass

# Compute moving average to smooth data
print("Computing moving avg...")

mov_avg_power_data = np.convolve(power_data[::SUBSAMPLING], np.ones(AVG_WINDOW) / AVG_WINDOW, mode="valid")
mov_avg_zmncc_0 = np.convolve(zmncc_results[0][::SUBSAMPLING], np.ones(AVG_WINDOW) / AVG_WINDOW, mode="valid") 
mov_avg_sad_0 = np.convolve(sad_results[0][::SUBSAMPLING], np.ones(AVG_WINDOW) / AVG_WINDOW, mode="valid") 

date = date_data[SKIP + SUBSAMPLING : len(date_data) - SKIP - SUBSAMPLING : SUBSAMPLING]

print(f"Len date: {len(date_data[SKIP + SUBSAMPLING : len(date_data) - SKIP - SUBSAMPLING : SUBSAMPLING])}   Len power: {len(mov_avg_power_data)}   Len zmncc: {len(mov_avg_zmncc_0)}   Len sad: {len(mov_avg_sad_0)} ")

print("Start plotting...")
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))

start_time = time.time()
axes[0].plot(date, mov_avg_power_data, label="Household Active Power", color="blue")
axes[0].set_ylabel('kW')
axes[0].set_title("Household Active Power")
axes[0].grid(True, which="both")

axes[1].plot(date, [val * FILTER_LENGTH // SUBSAMPLING if val <= FILTER_LENGTH // SUBSAMPLING else 0 for val in range(len(date))], label="Filter uptrend", color="deepskyblue")
axes[1].set_ylabel(f'Filter 0 : uptrend {FILTER_LENGTH}')
axes[1].grid(True, which="both")

axes[2].plot(date, mov_avg_zmncc_0, label="ZMNCC filter", color="red")
axes[2].set_ylabel('ZMNCC filter 0')
axes[2].grid(True, which="both")
axes[2].axhline(y=0, color="black", linestyle=(0, (5, 10)), alpha=0.5)

min_nonzero_val = min(filter(lambda x: x != 0, sad_results[0]))
max_val = max(mov_avg_sad_0)
axes[3].plot(date, mov_avg_sad_0, label="SAD filter", color="orange")
axes[3].set_ylabel('SAD filter 0')
axes[3].set_ylim(bottom=min_nonzero_val, top=max_val)
axes[3].grid(True, which="both")


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time elapsed to plot: {elapsed_time:.3f} s")

plt.savefig('./output_visualizer/data_visualizer.png')