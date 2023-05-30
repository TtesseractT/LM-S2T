'''
Model Tester from HF Fine tune Whisper Tiny Model

Author: Sabian Hibbs
University of Derby
United Kingdom, England

Licence MIT
'''

from transformers import pipeline
import gradio as gr

pipe = pipeline(model="sanchit-gandhi/whisper-tiny-hi")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small Hindi",
    description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()