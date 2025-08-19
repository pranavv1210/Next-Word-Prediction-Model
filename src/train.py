import numpy as np
import pandas as pd
from model import WordPredictionModel
from data_preprocessing import load_data, preprocess_data
from utils import save_model
import pickle

def train_model():
    # Load and preprocess the dataset
    data = load_data('sample.txt')
    X_train, y_train, X_test, y_test, unique_words, unique_word_index, WORD_LENGTH = preprocess_data(data)

    # Initialize the model
    model = WordPredictionModel(input_shape=(WORD_LENGTH, len(unique_words)), output_dim=len(unique_words))
    model.build_model()
    model.compile_model()

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Save the trained model weights
    save_model(model, 'keras_next_word_model.h5')

    # Save unique_words, unique_word_index, and WORD_LENGTH for inference
    with open("unique_words.pkl", "wb") as f:
        pickle.dump(unique_words, f)
    with open("unique_word_index.pkl", "wb") as f:
        pickle.dump(unique_word_index, f)
    with open("WORD_LENGTH.pkl", "wb") as f:
        pickle.dump(WORD_LENGTH, f)

if __name__ == "__main__":
    train_model()