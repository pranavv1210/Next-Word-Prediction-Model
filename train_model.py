import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
import re

# 1. Prepare your training data
# Define the path to your new dataset file
file_path = 'Sherlock Holmes.txt' # Make sure this matches your downloaded file's name and location

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()  # Read text and convert to lowercase
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory.")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# Basic text cleaning: remove special characters, numbers, and extra spaces
# Keeping punctuation might be useful for more complex models, but for now, let's keep it simple.
text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space

# If the text is very long, you might want to truncate it for faster training
# For example, to use only the first 1,000,000 characters (approx. 1MB of text):
# text = text[:1000000]

# Split the text into sequences based on spaces (individual words will be tokenized later)
# This creates a flat list of words to be used for n-gram sequence generation.
words = text.split()

# Join them back for tokenizer fitting to build a comprehensive vocabulary
full_text_for_tokenizer = " ".join(words)

# 2. Fit tokenizer
tokenizer = Tokenizer(oov_token="<unk>") # Add OOV token for out-of-vocabulary words
tokenizer.fit_on_texts([full_text_for_tokenizer])
vocab_size = len(tokenizer.word_index) + 1

print(f"Vocabulary size: {vocab_size}")

# 3. Prepare sequences for next-word prediction
# Create input sequences (n-grams) from the tokens
input_sequences = []
token_list = tokenizer.texts_to_sequences([full_text_for_tokenizer])[0]

# Define a fixed maximum sequence length for padding to manage memory.
# Sequences longer than this will be truncated; shorter ones will be padded.
# This is crucial for handling large text datasets.
fixed_max_sequence_len = 50 # Adjusted to a manageable length.

for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    
    # We only add sequences up to the fixed_max_sequence_len
    if len(n_gram_sequence) <= fixed_max_sequence_len:
        input_sequences.append(n_gram_sequence)
    else:
        # If the sequence exceeds the fixed_max_sequence_len,
        # take the last 'fixed_max_sequence_len' tokens
        input_sequences.append(n_gram_sequence[len(n_gram_sequence) - fixed_max_sequence_len:])

# Ensure input_sequences is not empty
if not input_sequences:
    print("No input sequences could be generated. This might happen with very short or malformed text. Exiting.")
    exit()

# Pad sequences to the fixed_max_sequence_len and split into X (input) and y (output)
# All sequences will now have 'fixed_max_sequence_len' tokens.
sequences = np.array(pad_sequences(input_sequences, maxlen=fixed_max_sequence_len, padding='pre'))

# X will be all tokens except the last one; y will be the last token (the target)
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)

# If X is empty, it means there's an issue with sequence generation
if X.shape[0] == 0:
    print("X (input data) is empty after sequence preparation. Exiting.")
    exit()

print(f"Fixed max sequence length used for padding: {fixed_max_sequence_len}")
print(f"Shape of X (input sequences): {X.shape}")
print(f"Shape of y (target words - one-hot encoded): {y.shape}")

# 4. Save tokenizer and sequence length
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
# Save the input length for prediction (which is X.shape[1])
with open('WORD_LENGTH.pkl', 'wb') as f:
    pickle.dump(X.shape[1], f) # This is the input_length for the Embedding layer

print("Tokenizer and WORD_LENGTH saved successfully.")

# 5. Build and train the model
# The input_length for the Embedding layer should be X.shape[1]
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=X.shape[1])) # Embedding dim 100
model.add(LSTM(150)) # LSTM units 150
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# You might need to adjust epochs based on your dataset size and desired accuracy.
epochs = 50 # Reduced epochs for faster demonstration
model.fit(X, y, epochs=epochs, verbose=1)

# 6. Save the model
model.save('keras_next_word_model.h5')
print("Model trained and saved successfully.")