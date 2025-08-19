import numpy as np
import pandas as pd
from model import WordPredictionModel
from data_preprocessing import load_data, preprocess_data
from utils import save_model

def train_model():
    # Load and preprocess the dataset
    data = load_data('path/to/dataset')
    X_train, y_train, X_test, y_test = preprocess_data(data)

    # Initialize the model
    model = WordPredictionModel()
    model.build_model()
    model.compile_model()

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Save the trained model weights
    save_model(model, 'path/to/save/model_weights.h5')

if __name__ == "__main__":
    train_model()