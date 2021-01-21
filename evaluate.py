from modules.polarity.cnn_model import CNNModel
from modules.preprocess import load_data, preprocess

if __name__ == '__main__':
    input_data, output = load_data("")
    pre_data = preprocess(input_data)
    # split data
    model = CNNModel()
    model.train(input_data, output)
    output = model.predict(input_data)

    # evaluate
    # cal P R F1
