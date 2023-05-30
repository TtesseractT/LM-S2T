'''
Model Tester from HF Fine tune Whisper Tiny Model

Author: Sabian Hibbs
University of Derby
United Kingdom, England

Licence MIT
'''

from transformers import pipeline
import gradio as gr

pipe = pipeline(model="LM-S2T-TINY-2")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper LM-S2T-TINY-2",
    description="Realtime demo for Englidh speech recognition using a fine-tuned Whisper tiny model.",
)

iface.launch()