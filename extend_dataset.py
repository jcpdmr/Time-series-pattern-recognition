def append_file_content(file_name, num_copies):
    # Read the content of the file
    with open(file_name, 'r') as file:
        content = file.read()
    
    # Open the file in append mode
    with open(file_name, 'a') as file:
        # Append the content of the file to the end for the specified number of times
        for _ in range(num_copies):
            file.write(content + '\n')

file_name = "input_data/input.txt"
num_copies = 1
append_file_content(file_name, num_copies)
