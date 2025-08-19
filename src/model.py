class WordPredictionModel:
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.model = self.build_model()

    def build_model(self):
        from tensorflow.keras import layers, models

        model = models.Sequential()
        model.add(layers.Embedding(self.vocab_size, self.embedding_dim))
        model.add(layers.LSTM(self.rnn_units, return_sequences=True))
        model.add(layers.LSTM(self.rnn_units))
        model.add(layers.Dense(self.vocab_size, activation='softmax'))
        return model

    def compile_model(self, learning_rate=0.001):
        from tensorflow.keras.optimizers import Adam

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=Adam(learning_rate),
                           metrics=['accuracy'])