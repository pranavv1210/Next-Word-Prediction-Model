import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
import re

# Define file paths. This makes it easier to manage across different environments.
data_file_name = "next_word_predictor.txt"
model_dir = "model"

# Check if running in a Colab environment to adjust paths accordingly.
is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False

if is_colab:
    # In Colab, the file is in the root content directory.
    data_path = os.path.join("/", "content", data_file_name)
    model_save_path = os.path.join("/", "content", model_dir)
else:
    # On a local machine (like VS Code), the script is in 'backend/',
    # so the data file is one level up.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", data_file_name)
    model_save_path = os.path.join(script_dir, model_dir)

# Create the model directory if it doesn't exist
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# --- 1. Load and Preprocess Data ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

try:
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print(f"Error: '{data_file_name}' not found at {data_path}.")
    print("Please make sure the dataset file is in the correct location.")
    exit()

cleaned_text = clean_text(text)
corpus = cleaned_text.split("\n")

# --- IMPORTANT CHANGE: Sample a smaller portion of the data ---
# This will drastically reduce memory usage and training time
# Adjust this number as needed to fit your machine's capabilities
sample_size = 50000 
if len(corpus) > sample_size:
    corpus = corpus[:sample_size]

# --- 2. Tokenization ---
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

max_sequence_len = max([len(x.split()) for x in corpus]) if corpus else 1

tokenizer_config = tokenizer.to_json()
tokenizer_data = json.loads(tokenizer_config)
tokenizer_data['config']['max_sequence_len'] = max_sequence_len

with open(os.path.join(model_save_path, "tokenizer.json"), 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_data, indent=4))

# --- 3. Create Input Sequences ---
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# --- 4. Build and Train the LSTM Model ---
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Starting model training...")
# For demonstration, we'll use a small number of epochs.
# For a real model, you would use more epochs and a larger dataset.
model.fit(xs, ys, epochs=10, verbose=1)

# --- 5. Save the Model ---
model.save(os.path.join(model_save_path, "lstm_model.h5"))

print("\nTraining complete.")
print(f"Tokenizer saved to: {os.path.join(model_save_path, 'tokenizer.json')}")
print(f"Model saved to: {os.path.join(model_save_path, 'lstm_model.h5')}")