import os

def append_file_content(source_file_name, destination_file_name, num_copies):
    # Read the content of source file
    with open(source_file_name, 'r') as file:
        content = file.readlines()[1:]  # Skipping the first line, because it contains the header row
    
    # Open the desetination file in append mode
    with open(destination_file_name, 'a') as file:
        # Append the content of the source file to the end for the specified number of times
        for _ in range(num_copies):
            file.writelines(content)

def get_num_lines(file_name):
    n_lines = 0
    with open(file_name, 'r') as file:
    # Read the number of lines in the file
        lines = file.readlines()
    if lines:
        n_lines = len(lines)      
    print(f"Total number of lines: {n_lines}")

def create_file_if_not_exists(file_name):
    # Check if file exists
    if not os.path.exists(file_name):
        # If it doeesn't exist, create the file
        with open(file_name, 'w'):
            pass

src_file_name = "input_data/household_power_consumption.txt"
dest_file_name = "input_data/input.txt"
num_copies = 10

if __name__ == "__main__":
    create_file_if_not_exists(dest_file_name)
    get_num_lines(dest_file_name)
    print("Extending dataset...")
    append_file_content(src_file_name, dest_file_name, num_copies)
    print("Done!")
    get_num_lines(dest_file_name)
