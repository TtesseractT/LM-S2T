# This script splits a directory of WAV files into 10 subdirectories.

import os
import shutil
import math
from tqdm import tqdm

# Specify the source directory containing the WAV files
source_directory = 'path_to_dataset'

# Create 10 target directories
target_directories = [os.path.join(source_directory, f'folder_{i+1}') for i in range(10)]
for directory in target_directories:
    os.makedirs(directory, exist_ok=True)

# Get a list of all WAV files in the source directory
wav_files = [file for file in os.listdir(source_directory) if file.endswith('.wav')]

# Calculate the number of files to move per target directory
files_per_directory = math.ceil(len(wav_files) / 10)

# Move the files to the target directories
with tqdm(total=len(wav_files), desc="Splitting files") as pbar:
    for i, file in enumerate(wav_files):
        source_path = os.path.join(source_directory, file)
        target_directory = target_directories[i // files_per_directory]
        target_path = os.path.join(target_directory, file)
        shutil.move(source_path, target_path)
        pbar.update(1)

print("Splitting complete!")
