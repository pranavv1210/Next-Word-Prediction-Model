import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    # Load text file and return as string
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def preprocess_data(data, word_length=5):
    # Tokenize words
    words = data.lower().split()
    unique_words = sorted(list(set(words)))
    unique_word_index = {w: i for i, w in enumerate(unique_words)}
    WORD_LENGTH = word_length

    prev_words = []
    next_words = []
    for i in range(len(words) - WORD_LENGTH):
        prev_words.append(words[i:i + WORD_LENGTH])
        next_words.append(words[i + WORD_LENGTH])

    X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
    Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
    for i, each_words in enumerate(prev_words):
        for j, each_word in enumerate(each_words):
            X[i, j, unique_word_index[each_word]] = 1
        Y[i, unique_word_index[next_words[i]]] = 1

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test, unique_words, unique_word_index, WORD_LENGTH

def split_data(data, test_size=0.2):
    return train_test_split(data, test_size=test_size, random_state=42)