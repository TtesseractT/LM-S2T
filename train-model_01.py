import os, torch, time
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Dataset
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor, TrainingArguments, Trainer
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

# pip install pandas soundfile datasets transformers numpy

# Define a function to preprocess audio and text
def preprocess_data(data):
    # Read the audio data from the file system
    audio_data = []
    for _, row in data.iterrows():
        audio_path = os.path.join("/Users/sabianhibbs/Desktop/Speech-Model-Training/clips", row["path"])
        audio, _ = sf.read(audio_path)
        audio_data.append(audio)

    # Convert the audio data to a NumPy array and pad it to the same length
    padded_audio = np.zeros((len(audio_data), max(len(audio) for audio in audio_data)))
    for i, audio in enumerate(audio_data):
        padded_audio[i, :len(audio)] = audio

    # Tokenize the text data
    text_data = [row["sentence"] for _, row in data.iterrows()]
    text_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    tokenized_text = text_tokenizer(text_data)

    # Return a tuple of audio data, text data, and tokenized text
    return padded_audio, tokenized_text

# Load the training data
data = load_dataset("common_voice", "en", split="train")

# Load the evaluation data
eval_dataset = load_dataset("common_voice", "en", split="test")

# Preprocess the training data
train_dataset = train_dataset.map(preprocess_data, batched=True, batch_size=8, remove_columns=["audio", "text"])

# Split the training data into training and validation sets
train_dataset, valid_dataset = train_dataset.train_test_split(test_size=0.1)

# Preprocess the evaluation data
eval_dataset = eval_dataset.map(preprocess_data, remove_columns=eval_dataset.column_names)

# Define the tokenizer and processor
text_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Define the model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def pad_audio(audio, max_length, padding_value=0.0):
    padded_audio = np.zeros(max_length)
    padded_audio[: len(audio)] = audio
    return padded_audio

def data_collator(features):
    max_length_input = max([len(f["input_values"]) for f in features])
    max_length_labels = max([len(f["labels"]) for f in features])
    
    padded_input_values = torch.tensor([pad_audio(f["input_values"], max_length_input) for f in features])
    padded_labels = torch.tensor([pad_audio(f["labels"], max_length_labels, padding_value=-100) for f in features], dtype=torch.long)

    return {"input_values": padded_input_values, "labels": padded_labels}

# Define a function to tokenize and process the dataset
def prepare_data(batch):
    max_length = max([len(audio) for audio in batch["audio"]])
    padded_audio = [pad_audio(audio, max_length) for audio in batch["audio"]]
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, do_normalize=True, return_attention_mask=True
    )
    input_values = feature_extractor(padded_audio, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
    labels = text_tokenizer(batch["text"], padding="longest", return_tensors="pt").input_ids
    return {"input_values": input_values, "labels": labels}

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=200,
)

# Create a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=text_tokenizer,
)

# Train the model 
trainer.train()

# Save the model
model.save_pretrained("./results")

# Evaluate the model
results = trainer.evaluate(eval_dataset)
print(results)
