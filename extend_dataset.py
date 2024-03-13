def append_file_content(source_file_name, destination_file_name, num_copies):
    # Read the content of the file
    with open(source_file_name, 'r') as file:
        content = file.read()
    
    # Open the file in append mode (it creates the file if it doesn't exist)
    with open(destination_file_name, 'a') as file:
        # Append the content of the file to the end for the specified number of times
        for _ in range(num_copies):
            file.write(content + '\n')

def get_num_lines(file_name):
    with open(file_name, 'r') as file:
    # Read the number of lines in the file
        lines = file.readlines()
    print(f"Total number of lines: {len(lines)}")

dest_file_name = "input_data/input.txt"
src_file_name = "input_data/household_power_consumption.txt"
num_copies = 1

if __name__ == "__main__":
    get_num_lines(dest_file_name)
    print("Extending dataset...")
    append_file_content(src_file_name, dest_file_name, num_copies)
    print("Done!")
    get_num_lines(dest_file_name)
