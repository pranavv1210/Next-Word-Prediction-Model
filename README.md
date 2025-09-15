# Next-Word Prediction Model

### A Deep Learning Web Application for Next-Word Prediction

-----

### 📝 Project Overview

This project is a minimalist, full-stack web application that predicts the next word as a user types. The application demonstrates the power of **pre-trained deep learning models** for **natural language processing (NLP)**. The core of the application is a **GPT-2** model from the Hugging Face `transformers` library, which is specifically designed for text generation. The front end provides a clean, user-friendly interface that communicates with a lightweight Python backend to provide real-time suggestions.

-----

### ✨ Features

  * **Real-time Predictions:** Get instant next-word suggestions as you type.
  * **Minimalist UI:** A clean and simple user interface that keeps the focus on the core functionality.
  * **Intelligent Suggestions:** The pre-trained GPT-2 model is highly effective at understanding context and providing coherent predictions.
  * **Simple Interaction:** Predicted words are displayed as "ghost text" and can be accepted with a simple key press (e.g., Tab).
  * **Scalable Architecture:** The project's structure is designed for easy expansion, allowing for future improvements and feature additions.

-----

### 🛠️ Technologies Used

  * **Backend:**
      * **Python:** The primary language for the backend logic.
      * **Flask:** A micro-web framework used to create a simple API for the model.
      * **Hugging Face `transformers`:** The library used to load and run the pre-trained GPT-2 model.
      * **PyTorch:** The deep learning framework required by the `transformers` library to run the model.
  * **Frontend:**
      * **HTML:** For the basic structure of the web page.
      * **CSS:** For styling the minimalist UI.
      * **JavaScript:** To handle user input and asynchronous communication with the backend API.

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
│   ├── model/                     # The pre-trained model files are stored here
│   └── venv/                      # Python virtual environment
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
git clone https://github.com/pranavv1210/Next-Word-Prediction-Model.git
cd Next-Word-Prediction-Model
```

#### Step 2: Set Up the Backend

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  (Optional but Recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    ```
3.  Activate the new virtual environment:
    ```bash
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```
4.  Install the required Python libraries:
    ```bash
    pip install Flask flask-cors transformers torch
    ```

#### Step 3: Run the Application

1.  Make sure you are in the `backend` directory and your virtual environment is active.
2.  Start the Flask server:
    ```bash
    python app.py
    ```
    The server will start running on `http://localhost:5000`. The first time you run this, it will download the pre-trained model, which may take a few minutes.
3.  Open the `frontend/index.html` file in your web browser. You can simply double-click the file to open it.

The web application should now be live and ready for you to use\!

-----

### 🤝 Contributing

This project is a personal portfolio piece. However, feel free to fork the repository, experiment, and share your suggestions.

-----

### 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.