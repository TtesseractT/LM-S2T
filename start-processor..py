import os
import os
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
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

def get_mp3_files(input_folder):
    mp3_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.mp3'):
                input_path = os.path.join(root, file)
                wav_path = os.path.splitext(input_path)[0] + '.wav'
                if not os.path.exists(wav_path):
                    mp3_files.append(input_path)
    return mp3_files

def convert_file(input_path):
    output_path = os.path.splitext(input_path)[0] + '.wav'
    command = f"ffmpeg -loglevel error -i \"{input_path}\" -acodec pcm_s16le -ac 1 -ar 16000 \"{output_path}\""
    subprocess.run(command, shell=True, check=True)
    os.remove(input_path)

def convert_to_wav(input_folder):
    mp3_files = get_mp3_files(input_folder)
    num_processes = cpu_count()  # Number of available processors
    
    with Pool(num_processes) as p:
        list(tqdm(p.imap(convert_file, mp3_files), total=len(mp3_files), desc="Converting MP3 to WAV", unit="file"))

input_folder = "path/to/input/folder"  # Set the variable to the desired location

convert_to_wav(input_folder)

with open('validated.tsv', 'r') as input_file, open('output_file.tsv', 'w') as output_file:
    for line in input_file:
        new_line = line.replace('.mp3', '.wav')
        output_file.write(new_line)

with open('validated.tsv', 'r') as input_file, open('output_file.tsv', 'w') as output_file:
    for line in input_file:
        new_line = line.replace('.mp3', '.wav')
        output_file.write(new_line)
