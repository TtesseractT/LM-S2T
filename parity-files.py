# This file removes the non validated audio 
# filed from the dataset folder 'clips' parsed 
# with the validated.tsv file.

import os

# Path to your audio dataset folder
dataset_folder = 'clips'

# Path to the validated.tsv file
validated_file_path = 'validated.tsv'

# Read the validated file and extract the audio file names
referenced_files = set()
with open(validated_file_path, 'r', encoding='utf-8') as validated_file:
    # Skip the header row
    next(validated_file)
    for line in validated_file:
        line_parts = line.strip().split('\t')
        file_name = line_parts[1]  # Extract the 'path' column
        referenced_files.add(file_name)

# Get the list of all audio files in the dataset folder
all_files = set(os.listdir(dataset_folder))

# Get the set of unreferenced audio files
unreferenced_files = all_files - referenced_files

# Delete the unreferenced audio files
for file_name in unreferenced_files:
    file_path = os.path.join(dataset_folder, file_name)
    os.remove(file_path)
    #print(f"Deleted: {file_path}")
