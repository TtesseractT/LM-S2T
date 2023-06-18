from transformers import pipeline
import torch
import argparse

parser = argparse.ArgumentParser(description='automatic speech recognition')
parser.add_argument('model_path', type=str, help='The model folder path')

# Optional positional argument
parser.add_argument('audio_path', type=str, help='The audio sample path')
args = parser.parse_args()

model_path = args.model_path

# transfer model

pipe = pipeline("automatic-speech-recognition", model=model_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


tokenizer = pipe.tokenizer
def transcribe(audio):
    out = pipe(audio)
    return out["text"]

import librosa as lr
def LoadAudio(x):
    x, sr = lr.load(x, sr=16000)
    return x

audio_path = args.audio_path
audio = LoadAudio(audio_path)
print(transcribe(audio))
