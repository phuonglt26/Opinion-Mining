import string

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential

from modules.evaluate import cal_sentiment_prf
from modules.five_folds_cross_validation import five_folds_cross_validation
from modules.polarity.cnn_model import PolarityCNNModel
from modules.polarity.dt_model import PolarityDTModel
from modules.polarity.kn_model import PolarityKNModel
from modules.polarity.lr_model import PolarityLRModel
from modules.polarity.nb_model import PolarityNBModel
from modules.polarity.rf_model import PolarityRFModel
from modules.polarity.svm_model import PolaritySVMModel
from modules.preprocess import preprocess, load_polarity_data


if __name__ == '__main__':
    NUM_OF_ASPECTS = 6
    tp = []
    fp = []
    fn = []
    for aspectId in range(NUM_OF_ASPECTS):
        inputs, outputs = load_polarity_data('data/raw_data/mebe_tiki.csv', aspectId)
        inputs = preprocess(inputs)
        model = PolarityDTModel()
        Sequential().compile()

        # five_folds_cross_validation(inputs, outputs, model, aspectId=aspectId)
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=14)
        model.train(X_train, y_train, aspectId)
        predicts = model.predict(X_test, aspectId)
        _tp, _fp, _fn, p, r, f1 = model.evaluate(y_test, predicts)
        # print(p,r,f1)
        tp.append(_tp)
        fp.append(_fp)
        fn.append(_fn)

    print('\tgiá\t\tdịch_vụ\t\tan_toàn\t\t\t\tchất_lượng\t\tship\t\tchính_hãng')
    # cal_sentiment_prf(tp, fp, fn, NUM_OF_ASPECTS, verbal=True)
    with open('data/output/evalDTModel.txt', 'w', encoding='utf-8') as f:
        f.write(cal_sentiment_prf(tp, fp, fn, NUM_OF_ASPECTS, verbal=True))




