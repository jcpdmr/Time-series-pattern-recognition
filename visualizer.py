import re
import os
import matplotlib.pyplot as plt
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

with open("input_data/household_power_consumption_short.txt", "r") as file:
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
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(70, 20))

start_time = time.time()
axes[0].plot(date_data[::720], power_data[::720], label="Household Active Power")
axes[0].set_ylabel('kW')
axes[0].set_title("Household Active Power")
axes[0].set_xticks(date_data[::133920])
axes[0].set_xticks(date_data[::44640], minor=True)
axes[0].grid(True, which="both")


axes[1].plot(date_data[::720], [data[1] for data in cross_correlation_data[0][::720]], label="...")
axes[1].set_ylabel('Uptrend 1 week')
axes[1].set_xticks(date_data[::133920])
axes[1].set_xticks(date_data[::44640], minor=True)
axes[1].grid(True, which="both")
axes[1].axhline(y=0, color="black", linestyle=(0, (5, 10)), alpha=0.5)

axes[2].plot(date_data[::720], [data[1] for data in cross_correlation_data[1][::720]], label="...")
axes[2].set_ylabel('Uptrend 4 weeks')
axes[2].set_xticks(date_data[::133920])
axes[2].set_xticks(date_data[::44640], minor=True)
axes[2].grid(True, which="both")
axes[2].axhline(y=0, color="black", linestyle=(0, (5, 10)), alpha=0.5)

axes[3].plot(date_data[::720], [data[1] for data in cross_correlation_data[2][::720]], label="...")
axes[3].set_ylabel('Uptrend 8 weeks')
axes[3].set_xticks(date_data[::133920])
axes[3].set_xticks(date_data[::44640], minor=True)
axes[3].grid(True, which="both")
axes[3].axhline(y=0, color="black", linestyle=(0, (5, 10)), alpha=0.5)

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