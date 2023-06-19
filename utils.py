from transformers import pipeline
import torch
import librosa as lr
import colorednoise as cn
import argparse
import numpy as np


def load_model(path):
    pipe = pipeline("automatic-speech-recognition", model=model_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    tokenizer = pipe.tokenizer()

    return pipe, tokenizer

def transcribe(audio, pipe):
    out = pipe(audio)
    return out["text"]
def LoadAudio(x):
    x, sr = lr.load(x, sr=16000)
    return x


def White_noise(x):
  noise_factor = 0.005
  white_noise = np.random.randn(len(x)) * noise_factor
  return  x + white_noise

def Pink_noise(x):
  pink_noise = cn.powerlaw_psd_gaussian(1, len(x))
  return x + pink_noise

def Brown_noise(x):
  brown_noise = cn.powerlaw_psd_gaussian(2, len(x))
  return x + brown_noise

def detect_silence(audio, threshold_db=-40, hop_length=512, duration=0.5):
    silence_segments = lr.effects.split(audio, top_db=threshold_db, hop_length=hop_length, duration=duration)
    return silence_segments

def calculate_metrics(true_words, predicted_words):
    total_words = 0
    error_words = 0
    correct_words = 0

    for true_sentence, predicted_sentence in zip(true_words, predicted_words):
        true_sentence_words = true_sentence
        predicted_sentence_words = predicted_sentence
        total_words += len(true_sentence_words)

        for true_word, predicted_word in zip(true_sentence_words, predicted_sentence_words):
            if true_word != predicted_word:
                error_words += 1
            else:
                correct_words += 1

    wer = (error_words / total_words) * 100
    accuracy = (correct_words / total_words) * 100

    return wer, accuracy


def main(folder_path, pipe, tokenizer, segment_duration=29, return_text=True)
    TS = []
    OUTS = []
    files = os.listdir(folder_path)
    for file in tqdm(files, desc="Transcribing files", unit="file"):
        audio_path = os.path.join(folder_path, file)
        
        # Load the audio
        audio = LoadAudio(audio_path)
        
        # Detect silence segments
        silence_segments = detect_silence(audio)
        
        # Transcribe each spoken-word segment
        transcripts = []
        for i, segment in enumerate(silence_segments):
            segment_transcript = transcribe(segment, pipe)
            transcripts.append(segment_transcript)
        
        # Merge the transcripts from all segments
        full_transcript = " ".join(transcripts)
        TS.append(full_transcript)
        
    # return the transcript 
    for full_transcript in TS:
        if not return_text:
            out = tokenizer._tokenize(full_transcript)
            f = [tokenizer._convert_token_to_id(o) for o in out]
            OUTS.append(np.asarray(f)) 
        else:
            OUTS.append(full_transcript)
    return OUTS, files


if __name__=="__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device name:", torch.cuda.get_device_name(0))  # Prints the name of the first GPU
        print("CUDA device count:", torch.cuda.device_count())  # Prints the number of available GPUs
    else:
        print("CUDA is not available.")
    parser = argparse.ArgumentParser(description='automatic speech recognition')
    parser.add_argument('model_path', type=str, help='The model folder path')
    parser.add_argument('data_path', type=str, help='The audio sample path')
    parser.add_argument('output_folder', type=str, help='The output folder path')
    args = parser.parse_args()

    model_path = args.model_path
    pipe, tokenizer = load_model(model_path)

    folder_path = args.data_path
    res, files = main(folder_path, pipe, tokenizer)

    output_folder = args.output_folder
    for r, f in zip(res, files):
        output_file = os.path.join(output_folder, f"{file}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(r)
