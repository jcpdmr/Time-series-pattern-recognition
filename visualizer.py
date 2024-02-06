import re
import os
import matplotlib.pyplot as plt

def custom_sort(string):
    # Extract the number from the string using regular expressions
    number = re.search(r'\d+', string).group()
    # Convert the number to an integer and return it for sorting
    return int(number)

os.chdir("./")

# Read close value
data_file = "input_data/NVDA_close.txt"
close_data = []

with open(data_file, 'r') as file:
    for i, line in enumerate(file, 1):
        value = float(line.strip())
        close_data.append((i, value))

# Read dates value
data_file = "input_data/NVDA_date.txt"
date_data = []

with open(data_file, 'r') as file:
    for i, line in enumerate(file, 1):
        value = line.strip()
        date_data.append(value)

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


# Plotting
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))

axes[0].plot(date_data, [data[1] for data in close_data], label="Close")
axes[0].set_ylabel('USD')
axes[0].set_xticks(date_data[::90])
axes[0].set_xticks(date_data[::15], minor=True)
axes[0].grid(True, which="both")


axes[1].plot(date_data, [data[1] for data in cross_correlation_data[3]], label="...")
axes[1].set_ylabel('Cross-correlation')
axes[1].set_xticks(date_data[::90])
axes[1].set_xticks(date_data[::15], minor=True)
axes[1].grid(True, which="both")

axes[2].plot(date_data, [i if i < 29 else 0 for i in range(0,1258) ], label="...")
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Cross-correlation')
axes[2].set_xticks(date_data[::90])
axes[2].set_xticks(date_data[::15], minor=True)
axes[2].grid(True, which="both")

plt.tight_layout()
plt.savefig('./output_visualizer/cross-corr2')