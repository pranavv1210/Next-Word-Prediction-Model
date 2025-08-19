# Next Word Prediction Model

An AI-powered web application that predicts the next word in real time as you type.

---

## Features

* Predicts the next word for your sentence in real time using a custom-trained **Keras LSTM model**.
* Click the suggestion button or press `Tab` to accept the prediction, or continue typing your own word.
* Modern, responsive frontend for a smooth user experience.
* Easy to set up and run locally.

---

## Requirements

* Python 3.7+
* pip

---

## Installation

1.  **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd Next-Word-Prediction-Model/next-word-prediction
    ```

2.  **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset

This model is designed to be trained on a large, general English text corpus to provide more meaningful predictions.

**To use your own dataset:**
1.  Obtain a large plain text file (e.g., `large_corpus.txt` from Project Gutenberg or similar sources).
2.  Place this file in the `next-word-prediction/` directory.

---

## Model Architecture

The core of the prediction engine is a **Keras Sequential model** featuring:
* An **Embedding layer** for word representations.
* **Stacked LSTM layers** (`return_sequences=True` for intermediate layers) to capture complex sequential dependencies in text.
* A final **Dense layer with Softmax activation** to output word probabilities.

The model is trained to predict the next word given a sequence of preceding words.

---

## Training the Model

By default, the `train_model.py` script expects a file named `large_corpus.txt`. Ensure your chosen dataset is named this way or update the `file_path` variable in `train_model.py`.

1.  **Prepare your dataset:** Place your large text file (e.g., `large_corpus.txt`) in the root of the `next-word-prediction/` directory.
2.  **Run the training script:** This process will read your dataset, preprocess it, train the Keras model, and save the necessary files (`keras_next_word_model.h5`, `tokenizer.pkl`, `WORD_LENGTH.pkl`).
    ```bash
    python train_model.py
    ```
    *Note: This step can take a considerable amount of time depending on the dataset size and your system's specifications.*

---

## Running the App

After successfully training your model (or if using pre-trained files), follow these steps:

### 1. Start the Backend (Flask API)

Open your terminal in the `next-word-prediction/` directory and run:
```bash
python app.py
````

  - The backend will run at `http://127.0.0.1:5000`. Keep this terminal window open.

### 2\. Serve the Frontend

In a **new** terminal window (keep the backend running), navigate to the `next-word-prediction/` directory and run:

```bash
python -m http.server 8000
```

  - Open your browser and go to: [http://localhost:8000/index.html](https://www.google.com/search?q=http://localhost:8000/index.html)

-----

## Usage

  - Start typing a sentence in the input box.
  - The predicted next word will appear as a blue button below.
  - Click the button or press `Tab` to insert the suggestion.
  - You can also ignore the suggestion and type your own word.

-----

## Project Structure

```
next-word-prediction/
│
├── app.py                     # Flask backend for predictions
├── index.html                 # Frontend user interface
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── train_model.py             # Script to train the Keras model
├── large_corpus.txt           # (Your chosen dataset, e.g., from Project Gutenberg)
│
├── keras_next_word_model.h5   # (Generated) The trained Keras model weights
├── tokenizer.pkl              # (Generated) Tokenizer fitted on the training data
├── WORD_LENGTH.pkl            # (Generated) Stores the fixed input sequence length
│
├── notebooks/                 # (Optional) Jupyter notebooks for exploration
│   └── exploration.ipynb
└── src/                       # (Optional) Source code for data processing, model, etc.
    ├── data_preprocessing.py
    ├── model.py
    ├── predict.py
    ├── train.py
    └── utils.py
```

-----

## Troubleshooting

  * **"Error: The file 'large\_corpus.txt' was not found."**: Ensure your dataset file is in the same directory as `train_model.py` and its name exactly matches the `file_path` variable in the script (case-sensitive).
  * **Memory Errors during training (`numpy.core._exceptions._ArrayMemoryError`)**: Your dataset might be too large or contain extremely long sentences. The `train_model.py` already includes a `fixed_max_sequence_len = 50` to mitigate this. If it persists, consider truncating your `large_corpus.txt` to a smaller size (e.g., `text = text[:5000000]`) or reducing `fixed_max_sequence_len`.
  * **No prediction appears / "Prediction failed. Is the backend running?"**: Make sure both the backend (`python app.py`) and frontend (`python -m http.server 8000`) servers are running in separate terminal windows. Check the browser console and backend terminal for any error messages.
  * **Model doesn't provide "meaningful" predictions**: This often indicates that the training dataset was either too small, not diverse enough, or the model needs more training epochs. Consider increasing the size and variety of your `large_corpus.txt` and re-running `train_model.py`.

-----

## License

This project is for educational purposes.

```
```