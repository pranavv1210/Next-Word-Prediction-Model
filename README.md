# Next Word Prediction Model

This project implements a Next Word Prediction model using deep learning techniques. The model is designed to predict the next word in a sequence based on the input provided. It utilizes various machine learning and deep learning libraries to preprocess data, build and train the model, and make predictions.

## Project Structure

```
next-word-prediction
├── src
│   ├── data_preprocessing.py    # Functions for loading and preprocessing the dataset
│   ├── model.py                 # Defines the neural network architecture
│   ├── train.py                 # Responsible for training the model
│   ├── predict.py               # Functions for making predictions
│   └── utils.py                 # Utility functions for various tasks
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── notebooks
    └── exploration.ipynb        # Jupyter notebook for exploratory data analysis
```

## Installation

To set up the project, clone the repository and install the required dependencies. You can do this by running the following commands:

```bash
git clone <repository-url>
cd next-word-prediction
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Use the `data_preprocessing.py` file to load and preprocess your dataset. This includes tokenization and padding sequences.

2. **Model Training**: Train the model by running the `train.py` script. This will utilize the `WordPredictionModel` class from `model.py` to build and compile the model, and then train it on the preprocessed data.

3. **Making Predictions**: After training, you can make predictions using the `predict.py` file. Call the `predict_next_word` function with an input sequence to get the predicted next word.

4. **Exploratory Data Analysis**: Use the Jupyter notebook located in the `notebooks` directory for any exploratory data analysis and visualizations related to your dataset.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.