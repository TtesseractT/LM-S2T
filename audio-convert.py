import os
import subprocess
import sys
from tqdm import tqdm
from multiprocessing import Pool

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

def convert_to_wav(input_folder, num_processes):
    mp3_files = get_mp3_files(input_folder)
    
    with Pool(num_processes) as p:
        list(tqdm(p.imap(convert_file, mp3_files), total=len(mp3_files), desc="Converting MP3 to WAV", unit="file"))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_mp3_to_wav.py <input_folder> [num_processes]")
        sys.exit(1)

    input_folder = sys.argv[1]
    num_processes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    convert_to_wav(input_folder, num_processes)
