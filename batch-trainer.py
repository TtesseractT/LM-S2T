import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

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
  x, sr = lr.load(f"clips/{x}".replace('mp3', 'wav'), sr=21000)
  audio_list.append(x)

X = np.array(pad_sequences(audio_list))

y = outs
print(y.shape)
print(X[0].shape)

X = np.expand_dims(X, -1)

np.save("X.npy", X)
np.save("y.npy", y)


def create_transformer_model(input_shape, output_dim, num_heads, ff_dim, num_blocks):
    inputs = keras.Input(shape=input_shape)

    # Transformer blocks
    x = inputs
    for _ in range(num_blocks):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=input_shape[1]
        )(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        # Add & Norm
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward neural network
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dense(input_shape[1])(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        # Add & Norm
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # LSTM layer
    x = layers.Bidirectional(layers.LSTM(64))(x)

    # Output layer
    outputs = layers.Dense(output_dim)(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
input_shape = (2110, 1)
output_dim = 16
num_heads = 2
ff_dim = 32
num_blocks = 3

model = create_transformer_model(input_shape, output_dim, num_heads, ff_dim, num_blocks)
model.summary()

model.compile(keras.optimizers.Adam(lr=0.001), 'categorical_crossentropy', metrics=["acc"])

# Assume we have some data in numpy arrays `train_data` and `train_labels`
# Convert them into a tf.data.Dataset object
train_dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Shuffle and batch the dataset
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Now fit the model
model.fit(train_dataset, epochs=10)
