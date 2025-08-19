def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def save_model(model, file_path):
    model.save(file_path)

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy