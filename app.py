from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)
CORS(app)

# --- Configuration for your Keras model ---
# It's highly recommended to use environment variables for API keys in production.
# The hardcoded Gemini API key is removed as it's not needed for the Keras model.

# Load the pre-trained Keras model
# Ensure the path is correct if your file structure is different on deployment
try:
    model = load_model('keras_next_word_model.h5')
    print("Keras model loaded successfully.")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    model = None  # Set model to None if loading fails

# Load the tokenizer
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

# Load the max sequence length (WORD_LENGTH)
try:
    with open('WORD_LENGTH.pkl', 'rb') as f:
        max_sequence_len = pickle.load(f)
    print(f"Max sequence length (WORD_LENGTH) loaded: {max_sequence_len}")
except Exception as e:
    print(f"Error loading WORD_LENGTH: {e}")
    max_sequence_len = None


def get_next_word_keras(text):
    if not model or not tokenizer or max_sequence_len is None:
        print("Model, tokenizer, or max_sequence_len not loaded. Cannot predict.")
        return ""

    # Convert the input text to a sequence of tokens
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Pad the sequence to the maximum length expected by the model
    # The model was trained with input_length = X.shape[1] which is `max_sequence_len - 1`
    # So, we slice the padded input to match the expected input shape of the model.
    padded_token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')[0]
    
    # Take all but the last element for prediction, as the last element was the target during training.
    model_input = np.array([padded_token_list[:-1]]) 

    # Predict the next word's probability distribution
    # Using verbose=0 to suppress prediction progress logs
    predicted_probs = model.predict(model_input, verbose=0)

    # Get the index of the word with the highest probability
    predicted_word_index = np.argmax(predicted_probs)

    # Convert the index back to a word
    if predicted_word_index in tokenizer.index_word:
        return tokenizer.index_word[predicted_word_index]
    else:
        return ""

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    # Use the Keras prediction function
    next_word = get_next_word_keras(text)
    return jsonify({"next_word": next_word})

if __name__ == "__main__":
    # Ensure the Flask app is only run if the model and tokenizer loaded successfully.
    if model is not None and tokenizer is not None and max_sequence_len is not None:
        app.run(debug=True)
    else:
        print("Flask app cannot start due to missing model, tokenizer, or WORD_LENGTH file.")