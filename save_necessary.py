# Define the input and output file names
input_file = 'check_moldesign.txt'
output_file = 'filtered_moldesign.txt'

# Define the prefixes to search for
prefixes = ('#####', '?????', '******', '------', '%%%%%%')

# Open the input file for reading and the output file for writing
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Loop through each line in the input file
    for line in infile:
        # Check if the line starts with any of the specified prefixes
        if line.startswith(prefixes):
            # Write the matching line to the output file
            outfile.write(line)

print(f"Filtered lines have been written to {output_file}.")
