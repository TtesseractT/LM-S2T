# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_VS1MmWUwqvmDY3TIxvl2sxzAFPJl7d3
"""

import pandas as pd
import numpy as np

df = pd.read_csv('validated.tsv', sep="\t")

df.head()

inps = df["path"] #X
outs = df["sentence"] #Y

def RMChars(y): # Remove the special charachters
    chars = [
    '!', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':',
    ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}',
    '~'
]
    return ''.join(ch for ch in y if ch not in chars)

outs = np.array([RMChars(x) for x in outs]).astype(str)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

# Create a tokenizer and fit it to the sentences.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(outs)

# Convert the sentences to sequences of integers using the tokenizer.
outs = tokenizer.texts_to_sequences(outs)

#print(outs)

#Pad sequences, having all same shape
outs = pad_sequences(outs)
#print(outs)

import librosa as lr

import tensorflow as tf
audio_list = []

from tqdm import tqdm
for x in tqdm(inps):
  x, sr = lr.load(f"clips/{x}".replace('mp3', 'wav'), sr=128)
  audio_list.append(x)

X = np.array(pad_sequences(audio_list))

y = outs
print(y.shape)
print(X[0].shape)

X = np.expand_dims(X, -1)

# Save X as a .txt file
#np.savetxt("X.txt", X)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_transformer_model(input_shape, output_dim, num_heads, ff_dim, num_blocks):
    inputs = keras.Input(shape=input_shape)

    # Positional encoding
    # Transformer blocks
    x = inputs
    for _ in range(num_blocks):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=input_shape[1]
        )(x, x)
        attention_output = layers.Dropout(0.1)(attention_output)

        # Feed-forward neural network
        ffn_output = layers.Dense(ff_dim, activation="relu")(attention_output)
        ffn_output = layers.Dense(input_shape[1])(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = ffn_output

    # LSTM layer
    x = layers.Bidirectional(layers.LSTM(64))(x)

    # Output layer
    outputs = layers.Dense(output_dim)(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
input_shape = (1351, 1)
output_dim = 16
num_heads = 2
ff_dim = 32
num_blocks = 3

model = create_transformer_model(input_shape, output_dim, num_heads, ff_dim, num_blocks)
model.summary()

model.compile(keras.optimizers.Adam(lr=0.001), 'categorical_crossentropy', metrics=["acc"])
model.fit(X, y, epochs=10)
