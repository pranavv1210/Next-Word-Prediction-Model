from keras.models import load_model
import numpy as np
from utils import preprocess_input

def predict_next_word(model_path, input_sequence, tokenizer):
    model = load_model(model_path)
    processed_input = preprocess_input(input_sequence, tokenizer)
    predictions = model.predict(processed_input)
    predicted_word_index = np.argmax(predictions, axis=-1)
    predicted_word = tokenizer.index_word[predicted_word_index[0]]
    return predicted_word