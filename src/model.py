from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

class WordPredictionModel:
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=self.input_shape))
        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('softmax'))

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def fit(self, X_train, y_train, validation_data, epochs=10, batch_size=32):
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )

    def save(self, filepath):
        self.model.save(filepath)