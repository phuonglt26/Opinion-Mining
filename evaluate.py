from models import Input
from modules.polarity.cnn_model import CNNModel
from sklearn.model_selection import train_test_split
from modules.preprocess import load_polarity_data_du_lich, preprocess
from modules.polarity.nb_model import HotelPolarityNBModel
from modules.evaluate import cal_aspect_prf

if __name__ == '__main__':
    evls = []
    for i in range(5):
        inputs, outputs = load_polarity_data_du_lich('data/raw_data/hotel_data.csv',i)
        inputs = preprocess(inputs, 'data/preprocess/vietnamese_stopwords.txt', 'data/preprocess/acronym.txt')
        print(inputs)
        # t = [outputs[i].aspects for i in range(10)]
        # print(t)
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.5, random_state=14)
        model = HotelPolarityNBModel()
        model.train(X_train, y_train)
        ts =[X_test[i].text for i in range(len(X_test))]
        predicts = model.predict(X_test)
        # for i in range(len(predicts)):
        #     print(X_test[i].text)
        #     print(predicts[i].scores)
        cal_aspect_prf(y_test, predicts, verbal=True)












    # scr = [predicts[i].scores for i in range(50)]
    # asp= [predicts[i].aspects for i in range(50)]
    # print(asp)
    # print(scr)

    # print('\tstaff, service\t\troom standard\t\tfood\t\t\t\tlocation, price\t\tfacilities')
    # cal_aspect_prf(y_test, predicts, num_of_aspect=1, verbal=True)

    # print(input_data[0].text)
    # print(output[0].aspects)
    # print(output[0].scores)

    # pre_data = preprocess(input_data)
    # # split data
    # model = CNNModel()
    # model.train(input_data, output)
    # output = model.predict(input_data)

    # evaluate
    # cal P R F1
