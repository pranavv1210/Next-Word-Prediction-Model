from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re

app = Flask(__name__)
CORS(app)

# Use a larger model for better predictions
generator = pipeline('text-generation', model='gpt2')

def predict_next_word(text):
    output = generator(text, max_new_tokens=3, num_return_sequences=1)
    generated = output[0]['generated_text']
    continuation = generated[len(text):].strip()
    # Only pick the first valid word (letters only)
    match = re.match(r"([A-Za-z']+)", continuation)
    next_word = match.group(1) if match else ""
    print(f"Input: '{text}' | Generated: '{generated}' | Next word: '{next_word}'")
    return next_word

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    next_word = predict_next_word(text)
    return jsonify({"next_word": next_word})

if __name__ == "__main__":
    app.run(debug=True)