import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data['text'].tolist()

def tokenize_text(texts, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def preprocess_data(texts, max_sequence_length=50):
    tokenizer = tokenize_text(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return padded_sequences, tokenizer

def split_data(data, test_size=0.2):
    return train_test_split(data, test_size=test_size, random_state=42)