import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import pipeline

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Load a well-trained text generation model
print("Loading pre-trained GPT-2 model...")
try:
    pipe = pipeline("text-generation", model="gpt2")
    print("Model loaded successfully!")
except Exception as e:
    pipe = None
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """API endpoint for prediction."""
    if not pipe:
        return jsonify({'error': 'Model failed to load'}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    seed_text = data['text']
    
    # We use the text generation pipeline to predict the next word
    predictions = pipe(seed_text, max_new_tokens=1, do_sample=True, top_k=50, return_full_text=False)
    
    # Extract the generated text, which is the new word/token
    predicted_text = predictions[0]['generated_text']
    
    # Clean up any extra spaces that might be at the beginning of the generated word
    predicted_word = predicted_text.strip()

    return jsonify({'prediction': predicted_word})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)