### A Deep Learning Web Application for Next-Word Prediction

-----

### 📝 Project Overview

This project is a minimalist, full-stack web application that predicts the next word as a user types. The application is built to demonstrate the power of **deep learning** and **natural language processing (NLP)**. The core of the application is a **Long Short-Term Memory (LSTM)** neural network trained on a large text corpus. The front end provides a clean, user-friendly interface that communicates with a lightweight Python backend to provide real-time suggestions.

-----

### ✨ Features

  * **Real-time Predictions:** Get instant next-word suggestions as you type.
  * **Minimalist UI:** A clean and simple user interface that keeps the focus on the core functionality.
  * **Intelligent Suggestions:** The LSTM model is capable of understanding long-term context, leading to more accurate and coherent predictions.
  * **Simple Interaction:** Predicted words are displayed as "ghost text" and can be accepted with a simple key press (e.g., Tab).
  * **Scalable Architecture:** The project's structure is designed for easy expansion, allowing for future improvements and feature additions.

-----

### 🛠️ Technologies Used

  * **Backend:**
      * **Python:** The primary language for the backend logic.
      * **Flask:** A micro-web framework used to create a simple API for the model.
      * **TensorFlow/Keras:** The deep learning framework used to build, train, and run the LSTM model.
      * **NumPy:** Essential for numerical operations and data manipulation.
  * **Frontend:**
      * **HTML:** For the basic structure of the web page.
      * **CSS:** For styling the minimalist UI.
      * **JavaScript:** To handle user input and asynchronous communication with the backend API.
  * **Dataset:**
      * **Kaggle:** A large public text dataset from Kaggle was used to train the language model. The specific dataset is [Link to the Kaggle dataset you choose].

-----

### 🧑‍💻 Team Members

  * **Pranav V**
  * **Sushmitha V**

-----

### 📦 Project Structure

```
next-word-predictor/
├── backend/
│   ├── app.py                     # Flask application and API endpoint
│   ├── requirements.txt           # Python dependencies
│   └── model/
│       ├── lstm_model.h5          # Trained LSTM model (binary file)
│       └── tokenizer.json         # Tokenizer configuration (JSON)
├── frontend/
│   ├── index.html                 # Main web page
│   ├── style.css                  # Stylesheet
│   └── script.js                  # Frontend logic and API calls
├── .gitignore                     # Specifies files to ignore in Git
└── README.md                      # This file
```

-----

### 🚀 Getting Started

Follow these instructions to set up and run the project locally.

#### Prerequisites

  * **Python 3.x** installed on your system.
  * **pip** (Python package installer).

#### Step 1: Clone the Repository

First, clone this repository to your local machine using git:

```bash
git clone https://github.com/YOUR_USERNAME/next-word-predictor.git
cd next-word-predictor
```

#### Step 2: Set Up the Backend

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  (Optional but Recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Add the Model and Tokenizer:** The trained model and tokenizer files are required to run the application. You must place your `lstm_model.h5` and `tokenizer.json` files inside the `backend/model/` directory.

#### Step 3: Run the Application

1.  Make sure you are in the `backend` directory.

2.  Start the Flask server:

    ```bash
    python app.py
    ```

    The server will start running on `http://localhost:5000`.

3.  Open the `frontend/index.html` file in your web browser. You can simply double-click the file to open it.

The web application should now be live and ready for you to use\!

-----

### 🤝 Contributing

This project is a personal portfolio piece. However, feel free to fork the repository, experiment, and share your suggestions.

-----

### 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
