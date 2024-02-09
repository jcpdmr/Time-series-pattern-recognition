import re
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from datetime import datetime

def custom_sort(string):
    # Extract the number from the string using regular expressions
    number = re.search(r'\d+', string).group()
    # Convert the number to an integer and return it for sorting
    return int(number)

os.chdir("./")

date_data : list[datetime] = []
power_data : list[float] = [] 

format = "%d/%m/%Y %H:%M:%S"

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
        date_data.append(date_time)
        power_data.append(power)



# Read all cross-correlation data
output_folder = "output_data"

# Sorting the list using custom_sort function as key
list_of_strings = os.listdir(output_folder)
files = sorted(list_of_strings, key=custom_sort)

cross_correlation_data = []

for cross_correlation_file in files:
    temp = []
    pattern = r"zn_cross_correlation\d+_filterlen(\d+).txt"
    filter_length = int(re.search(pattern=pattern, string=cross_correlation_file).group(1))
    offset = int(filter_length / 2)
    # print(f"File: {cross_correlation_file}, filterlen: {filter_length}, offset: {int(filter_length / 2)}")

    with open(os.path.join(output_folder, cross_correlation_file), 'r') as file:
        # Add zero padding at the beginning
        for i in range(offset):
            temp.append((i, 0))
        # Get data from file
        for i, line in enumerate(file, 1):
            value = float(line.strip())
            temp.append((i + offset, value))
            last_iter = i
        # Add zero padding at the end
        for i in range(offset - 1):
            temp.append((last_iter + i, 0))

    cross_correlation_data.append(temp)

print("Start plotting")

# Plotting

mpl.rcParams['path.simplify'] = True

mpl.rcParams['path.simplify_threshold'] = 0.0
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30, 15))

start_time = time.time()
axes[0].plot(date_data[::1440], power_data[::1440], label="Power")
axes[0].set_ylabel('MW?')
axes[0].set_xticks(date_data[::500000])
axes[0].set_xticks(date_data[::100000], minor=True)
# axes[0].grid(True, which="both")


axes[1].plot(date_data[::1440], [data[1] for data in cross_correlation_data[3][::1440]], label="...")
axes[1].set_ylabel('Cross-correlation')
axes[1].set_xticks(date_data[::500000])
axes[1].set_xticks(date_data[::100000], minor=True)
axes[1].grid(True, which="both")

# axes[2].plot(date_data, [i if i < 29 else 0 for i in range(0,1258) ], label="...")
# axes[2].set_xlabel('Date')
# axes[2].set_ylabel('Cross-correlation')
# axes[2].set_xticks(date_data[::90])
# axes[2].set_xticks(date_data[::15], minor=True)
# axes[2].grid(True, which="both")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time elapsed to plot: {elapsed_time} s")

# plt.tight_layout()
plt.savefig('./output_visualizer/cross-corr2')