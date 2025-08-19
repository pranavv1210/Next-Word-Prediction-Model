# Next Word Prediction (AI-powered)

This project is a web app for predicting the next word in a sentence using AI models. You can use either a custom-trained Keras model or a large language model API (like Gemini or GPT-2) for predictions.

---

## Features

- **Frontend:** Simple web interface for entering text and seeing next-word suggestions.
- **Backend:** Flask API that predicts the next word using a selected AI model.
- **Model Options:**
  - Custom Keras model (trained on your own data)
  - Hugging Face GPT-2 (no training required)
  - Google Gemini API (requires API key)

---

## Setup

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd Next\ Word\ Prediction\ Model/next-word-prediction
```

### 2. Install Dependencies

For Keras or GPT-2:
```sh
pip install flask flask-cors tensorflow numpy transformers torch
```

For Gemini API:
```sh
pip install flask flask-cors google-generativeai
```

---

## Usage

### **A. Using a Custom Keras Model**

1. **Train the model:**
   - Edit `train_model.py` with your own sentences.
   - Run:
     ```sh
     python train_model.py
     ```
   - This will generate `keras_next_word_model.h5`, `tokenizer.pkl`, and `WORD_LENGTH.pkl`.

2. **Start the backend:**
   ```sh
   python app.py
   ```

3. **Open `index.html` in your browser.**

---

### **B. Using Hugging Face GPT-2 (No Training Needed)**

1. Replace your `app.py` with the GPT-2 version (see earlier in this chat).
2. Install dependencies:
   ```sh
   pip install flask flask-cors transformers torch
   ```
3. Start the backend:
   ```sh
   python app.py
   ```

---

### **C. Using Google Gemini API**

1. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Replace your `app.py` with the Gemini version (see earlier in this chat).
3. Install dependencies:
   ```sh
   pip install flask flask-cors google-generativeai
   ```
4. Start the backend:
   ```sh
   python app.py
   ```

---

## Project Structure

```
next-word-prediction/
├── app.py
├── train_model.py
├── index.html
├── keras_next_word_model.h5
├── tokenizer.pkl
├── WORD_LENGTH.pkl
├── requirements.txt
└── README.md
```

---

## Notes

- **API keys are sensitive.** Never share them publicly.
- For best results, use a large and diverse dataset when training your own model.
- If using Gemini or GPT-2, you do not need to train a model.

---

## License

MIT License

---

## Credits