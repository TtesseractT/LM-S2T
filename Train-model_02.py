import os, torch, time
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Dataset
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor, TrainingArguments, Trainer
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

# pip install pandas soundfile datasets transformers numpy
max_retries = 2    # Maximum number of retries
retry_count = 0     # Counter for retries
delay_s = 5         # Amount of retries


while retry_count < max_retries:
    try:
        data = pd.read_csv("validated.tsv", sep="\t").to_dict("records")

        # Define a function to preprocess audio and text
        def preprocess_data(data):
            audio_data = []
            text_data = []
            for example in data:
                audio_path = os.path.join("/Users/sabianhibbs/Desktop/Speech-Model-Training/clips", example["path"])
                audio, _ = sf.read(audio_path)
                audio_data.append(audio)
                text_data.append(example["sentence"])
            return audio_data, text_data

        # Preprocess the data
        audio_data, text_data = preprocess_data(data)

        # Create a Hugging Face dataset
        dataset = Dataset.from_dict({"audio": audio_data, "text": text_data})

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

        # Tokenize and process the dataset
        dataset = dataset.map(prepare_data, batched=True, batch_size=8, remove_columns=["audio", "text"])

        # Split the dataset into training and validation sets
        train_dataset = dataset.train_test_split(test_size=0.1)["train"]
        valid_dataset = dataset.train_test_split(test_size=0.1)["test"]

        # Load test dataset
        test_dataset = load_dataset("common_voice", "en", split="test")

        # Preprocess test dataset
        test_dataset = test_dataset.map(preprocess_data, remove_columns=test_dataset.column_names)

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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            tokenizer=text_tokenizer,
        )

        # Train the model 
        trainer.train()
        break
    
    except Exception as e:
        # Exception handling code
        print(f"Attempt {retry_count + 1} failed: {str(e)}")
        retry_count += 1

        time.sleep(delay_s)

if retry_count == max_retries:
    print("Operation failed after maximum retries.")