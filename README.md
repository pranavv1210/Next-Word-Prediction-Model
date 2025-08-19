# Next Word Prediction Model

This project is an AI-powered web app that predicts the next word as you type. It uses a Python Flask backend with a GPT-2 language model and a simple HTML/JS frontend.

---

## Features

- Predicts the next word for your sentence in real time
- Click or press Tab to accept the suggestion, or type your own word
- Modern, responsive frontend
- Easy to run locally

---

## Requirements

- Python 3.7+
- pip

---

## Installation

1. **Clone the repository**  
   ```
   git clone <your-repo-url>
   cd Next-Word-Prediction-Model/next-word-prediction
   ```

2. **Install Python dependencies**  
   ```
   pip install -r requirements.txt
   ```

---

## Running the App

### 1. Start the Backend (Flask API)

```
python app.py
```
- The backend will run at `http://127.0.0.1:5000`.

### 2. Serve the Frontend

In a new terminal, run:
```
python -m http.server 8000
```
- Open your browser and go to: [http://localhost:8000/index.html](http://localhost:8000/index.html)

---

## Usage

- Start typing a sentence in the input box.
- The predicted next word will appear as a blue button below.
- Click the button or press `Tab` to insert the suggestion.
- You can also ignore the suggestion and type your own word.

---

## Customization

- **Model:**  
  By default, the backend uses `gpt2` for predictions. You can change the model in `app.py` for different results.
- **Your Own Model:**  
  If you have a fine-tuned Keras model, you can integrate it by modifying `app.py`.

---

## Project Structure

```
next-word-prediction/
│
├── app.py                # Flask backend
├── index.html            # Frontend
├── requirements.txt      # Python dependencies
├── README.md
├── keras_next_word_model.h5  # (Optional) Your own Keras model
├── sample.txt
├── unique_words.pkl
├── unique_word_index.pkl
├── WORD_LENGTH.pkl
├── notebooks/
│   └── exploration.ipynb
└── src/
    ├── data_preprocessing.py
    ├── model.py
    ├── predict.py
    ├── train.py
    └── utils.py
```

---

## Troubleshooting

- **No prediction appears:**  
  Make sure both backend and frontend servers are running. Check the browser console and backend terminal for errors.
- **CORS errors:**  
  The backend enables CORS by default. If you change ports or domains, update CORS settings in `app.py`.
- **Slow predictions:**  
  The first prediction may take a few seconds as the model loads.

---

## License

This project is for educational purposes.