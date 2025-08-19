import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Prepare your training data
sentences = [
    "hello how are you",
    "hi how are you",
    "hello there",
    "how are you doing",
    "hi there how are you",
    "hello how is everything",
    "how is your day",
    "hi how is your day",
    "Good morning, how are you?",
    "Hey there!",
    "How's it going?",
    "What's up?",
    "How have you been?",
    "Long time no see!",
    "Good to see you.",
    "How are things?",
    "Hope you're doing well.",
    "Nice to meet you.",
    "What's new?",
    "How's life treating you?",
    "Good afternoon!",
    "Good evening!",
    "How's your evening?",
    "What's good?",
    "How are you feeling today?",
    "It's good to see you again.",
    "How's everything going?",
    "How's work?",
    "How's school?",
    "What have you been up to?",
    "Pleasure to meet you.",
    "How's your family?",
    "Are you having a good day?",
    "Hello again!",
    "Just wanted to say hi."
]

# 2. Fit tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1

# 3. Prepare sequences for next-word prediction
sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        sequences.append(n_gram_seq)

# Pad sequences and split into X (input) and y (output)
max_seq_len = max([len(seq) for seq in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_seq_len, padding='pre'))
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)

# 4. Save tokenizer and sequence length
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('WORD_LENGTH.pkl', 'wb') as f:
    pickle.dump(X.shape[1], f)

# 5. Build and train the model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=X.shape[1]))
model.add(LSTM(64))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)

# 6. Save the model
model.save('keras_next_word_model.h5')