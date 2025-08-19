from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

genai.configure(api_key="AIzaSyDujJuxV5l1dj3VNz_79WPA-Wfjrc774vU")

# List available models for debugging
print("Available models:")
for m in genai.list_models():
    print(m.name)

# Use the correct model name from the list above
model = genai.GenerativeModel("models/gemini-pro")

def get_next_word(text):
    prompt = f"Given the text: '{text}', predict only the next word (do not repeat the input, do not add punctuation, do not explain, just the next word):"
    response = model.generate_content(prompt)
    next_word = response.text.strip().split()[0]
    return next_word

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    next_word = get_next_word(text)
    return jsonify({"next_word": next_word})

if __name__ == "__main__":
    app.run(debug=True)