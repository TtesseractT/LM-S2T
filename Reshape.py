# This script reshapes the TSV file and the 
# audio files to match the format of the LibriSpeech dataset.

import os
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

# Load the TSV data
df = pd.read_csv('validated.tsv', sep='\t')

# Create a new dataframe for the new TSV file
new_df = pd.DataFrame(columns=df.columns)

# Audio files directory
audio_dir = 'clips'

# New audio files directory
new_audio_dir = 'Audio-Clips'

# Create new_audio_dir if it doesn't exist
if not os.path.exists(new_audio_dir):
    os.makedirs(new_audio_dir)

# Loop over each row in the data
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Load the audio file
    audio = AudioSegment.from_wav(os.path.join(audio_dir, row['path']))
    
    # Split the sentence into words
    words = row['sentence'].split()
    
    # Split the audio file into chunks by silence
    chunks = split_on_silence(audio)
    
    # Ensure the number of audio chunks equals the number of words
    if len(words) == len(chunks):
        for i, chunk in enumerate(chunks):
            # Define the new file name
            new_file_name = os.path.join(new_audio_dir, row['path'].replace('.wav', f'-{i}.wav'))
            
            # Save the chunk as a new audio file
            chunk.export(new_file_name, format="wav")
            
            # Add a new row to the new dataframe
            new_row = row.copy()
            new_row['path'] = new_file_name
            new_row['sentence'] = words[i]
            new_df = new_df.append(new_row, ignore_index=True)

# Save the new dataframe as a new TSV file
new_df.to_csv('new_tsv_file.tsv', sep='\t', index=False)
