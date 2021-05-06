from sklearn.model_selection import train_test_split

from modules.evaluate import cal_sentiment_prf
from modules.polarity.chi2 import Polaritychi2Model
from modules.preprocess import preprocess, load_polarity_data

if __name__ == '__main__':
    NUM_OF_ASPECTS = 5
    f1_pos = []
    f1_neg = []
    for aspectId in range(NUM_OF_ASPECTS):
        inputs, outputs = load_polarity_data('data/raw_data/hotel_data.csv', aspectId)
        inputs = preprocess(inputs)
        model = Polaritychi2Model()

        # aspectName = ['giá', 'dịch_vụ', 'an_toàn', 'chất_lượng', 'ship', 'chính_hãng']
        aspectName = ['staff, service', 'room standard', 'food', 'location, price', 'facilities']
        # five_folds_cross_validation(inputs, outputs, model, aspectId=aspectId)
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
        model.train(X_train, y_train, aspectId)
        # file_path = 'save_model/Model_save/lr_model/{}'.format(aspectName[aspectId])
        # model.save(file_path, aspectId)
        # model.load(file_path, aspectId)
        predicts = model.predict(X_test, aspectId)
        _tp, _fp, _fn, _p, _r, _f1 = model.evaluate_pos(y_test, predicts)
        f1_pos.append(_f1)

        _tp, _fp, _fn, _p, _r, _f1 = model.evaluate_neg(y_test, predicts)
        f1_neg.append(_f1)
    print('tích cực')
    cal_sentiment_prf(f1_pos, NUM_OF_ASPECTS, verbal=True)
    print('tiêu cực')
    cal_sentiment_prf(f1_neg, NUM_OF_ASPECTS, verbal=True)


    # outputs = zip(X_test, predicts)
    # with open('data/output/RFModel_giá.csv', mode='w', encoding="utf-8") as file:
    #     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(['Text', 'giá'])
    #     for output in outputs:
    #         writer.writerow(['{}'.format(output[0].text), '{}'.format(output[1].scores)])

    # print('price,service,safety,quality,ship,authenticity, micro, macro')

    # with open('data/evaluate/phobert/evalRFModel.txt', 'w', encoding='utf-8') as f:
    #     f.write(cal_sentiment_prf(tp, fp, fn, p, r, f1, NUM_OF_ASPECTS, verbal=True))
