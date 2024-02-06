import re
import os
import matplotlib.pyplot as pyplot

# Read close value
data_file = "input_data/NVDA_close.txt"
close_data = []

with open(data_file, 'r') as file:
    for i, line in enumerate(file, 1):
        value = float(line.strip())
        close_data.append((i, value))

# Stampare i dati estratti
for idx, value in close_data:
    print(f"{idx}: {value}")

# Read all cross-correlation data
output_folder = "output_data"
files = os.listdir(output_folder)

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
        for i in range(offset):
            temp.append((last_iter + i, 0))

    cross_correlation_data.append(temp)


