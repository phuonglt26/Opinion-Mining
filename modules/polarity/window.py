import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models import PolarityOutput
from modules.models import Model


# WINDOW
class PolaritywindowModel(Model):
    def __init__(self):
        self.NUM_OF_ASPECTS = 6
        self.vocab = []
        labelVocab = ["giá", "dịch_vụ", "an_toàn", "chất_lượng", "ship", "chính_hãng"]
        for label in labelVocab:
            _vocab = []
            with open('data/vocab/mebe_shopee/label_{}_mebe_shopee.txt'.format(label), encoding="utf-8") as f:
                for l in f:
                    l = l.split(',')
                    flag_scores = {'giá': 0, 'dịch_vụ': 0, 'an_toàn': 0, 'chất_lượng': 0, 'ship': 0, 'chính_hãng': 0}

                    if float(l[1]) > flag_scores[label]:
                        _vocab.append(l)
            self.vocab.append(_vocab)
        self.models = [LogisticRegression() for _ in range(self.NUM_OF_ASPECTS)]
        # RandomForestClassifier
        # LogisticRegression
        # MultinomialNB
        # KNeighborsClassifier
        # DecisionTreeClassifier
        # SVC
    def _locating(self, inputs, aspectId):
        window = 10
        inputs_window = []
        label = ["giá", "dịch_vụ", "an_toàn", "chất_lượng", "ship", "chính_hãng"]
        aspectName = label[aspectId]
        flag_scores = {'giá': 20, 'dịch_vụ': 0, 'an_toàn': 8, 'chất_lượng': 8, 'chính_hãng': 9, 'ship': 7} #mebe_tiki
        # flag_scores = {'giá': 0, 'dịch_vụ': 0, 'an_toàn': 0, 'chất_lượng': 0, 'chính_hãng': 0, 'ship': 0}
        flag_score = flag_scores[aspectName]
        for ip in inputs:
            text = ip.text.split()   # tách các từ trong câu thành mảng
            scores = [float(v[1]) for v in self.vocab[aspectId] if v[0] in text] # điểm của các từ trong mảng từ trên tương ứng
            _max = max(np.array(scores)) # tính score cao nhất trong câu
            n = len(text)
            indexs = [i for i in range(n)]
            text_scores = zip(text, scores, indexs) # zip từ và score tương ứng
            arr = []
            if _max < flag_score:  # nếu score cao nhất nhỏ hơn ngưỡng đã chọn cho aspect
                if _max != 0.0:  # nếu score lớn nhất khác 0 (chỉ dùng cho trường hợp đã lọc vocab và có câu gán nhãn sai
                    #word_center = [i[0] for i in text_scores if i[1] == _max] # word center là từ có score lớn nhất
                    ind_word_centers = [i[2] for i in text_scores if i[1] == _max]  # tìm index của word center
                    words = []
                    for ind in ind_word_centers:
                        if n - ind - 1 < window and ind < window:
                            for i in text:
                                if i not in words:
                                    words.append(i)
                        elif n - ind - 1 < window:
                            for i in range(ind - window, n):
                                if i not in words:
                                    words.append(text[i])
                        elif ind < window:
                            for i in range(0, ind + window + 1):
                                if i not in words:
                                    words.append(text[i])
                        else:
                            for i in range(ind - window, ind + window + 1):
                                if i not in words:
                                    words.append(text[i])

                    inputs_window.append(words)
                else:
                    inputs_window.append(text)

            else:  # khi score max < ngưỡng đã chọn cho aspect
                ind_word_centers = [i[2] for i in text_scores if i[1] >= flag_score]  # tìm index của word center
                words = []
                for ind in ind_word_centers:
                    if n - ind - 1 < window and ind < window:
                        for i in text:
                            if i not in words:
                                words.append(i)
                    elif n - ind - 1 < window:
                        for i in range(ind - window, n):
                            if i not in words:
                                words.append(text[i])
                    elif ind < window:
                        for i in range(0, ind + window + 1):
                            if i not in words:
                                words.append(text[i])
                    else:
                        for i in range(ind - window, ind + window + 1):
                            if i not in words:
                                words.append(text[i])
                inputs_window.append(words)
        return inputs_window

    def _represent(self, inputs, aspectId):
        """
        :param list of models.Input inputs:
        :return:
        """
        inputs_locating = self._locating(inputs, aspectId)

        features = []
        for ip in inputs_locating:
            _features = [float(v[1]) if v[0] in ip else 0 for v in
                         self.vocab[aspectId]]
            # _features = [1 if v[0] in ip else 0 for v in
            #              self.vocab[aspectId]]
            features.append(_features)
        return np.array(features).astype(np.float)

    def train(self, inputs, outputs, aspectId):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        X = self._represent(inputs, aspectId)
        ys = [output.scores for output in outputs]

        self.models[aspectId].fit(X, ys)

    def save(self, path, aspectId):
        # save the model to disk
        pickle.dump(self.models[aspectId], open(path, 'wb'))

    def load(self, path, aspectId):
        # load the model from disk
        model = pickle.load(open(path, 'rb'))
        self.models[aspectId] = model

    def predict(self, inputs, aspectId):
        """
        :param inputs:
        :return:
        :rtype: list of models.AspectOutput
        """
        X = self._represent(inputs, aspectId)
        outputs = []
        predicts = self.models[aspectId].predict(X)
        for output in predicts:
            label = 'aspect{}'.format(aspectId) + (' -' if output == -1 else ' +')
            aspect = 'aspect{}'.format(aspectId)
            outputs.append(PolarityOutput(label, aspect, output))
        return outputs

    def evaluate_pos(self, y_test, y_predicts):
        tp = 0
        fp = 0
        fn = 0
        for g, p in zip(y_test, y_predicts):
            if g.scores == p.scores == 1:
                tp += 1
            elif g.scores == 1:
                fn += 1
            elif p.scores == 1:
                fp += 1
        if tp == 0 and fp == 0:
            print("khong bat duoc")
            p = 0
        else:
            p = tp / (tp + fp)
        # if tp == 0 and fn == 0:
        #     r = 0
        # else:
        r = tp / (tp + fn)
        if r == 0 and p == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        return tp, fp, fn, p, r, f1

    def evaluate_neg(self, y_test, y_predicts):
        tp = 0
        fp = 0
        fn = 0
        for g, p in zip(y_test, y_predicts):
            if g.scores == p.scores == -1:
                tp += 1
            elif g.scores == -1:
                fn += 1
            elif p.scores == -1:
                fp += 1
        if tp == 0 and fp == 0:
            print("khong bat duoc")
            p = 0
        else:
            p = tp / (tp + fp)
        # if tp == 0 and fn == 0:
        #     r = 0
        # else:
        r = tp / (tp + fn)
        if r == 0 and p == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        return tp, fp, fn, p, r, f1




    # def _locating(self, inputs, aspectId):
    #     window = 3
    #     inputs_window = []
    #     for ip in inputs:
    #         _features = [v[1] if v[0] in ip.text else 0 for v in
    #                      self.vocab[aspectId]]
    #         text = [v[0] if v[0] in ip.text else 0 for v in
    #                 self.vocab[aspectId]]
    #         _max = max(np.array(_features).astype(np.float))
    #         t = ip.text.split()
    #         if _max != 0.0:
    #             max_index_col = np.argmax(np.array(_features).astype(np.float), axis=0)
    #             max_vocab = text[max_index_col]
    #
    #             # print(max_vocab)
    #             center_vocab = ''
    #             for i in t:
    #                 if max_vocab in i:
    #                     center_vocab = i
    #             ind = t.index(center_vocab)
    #             n = len(t)
    #             arr = []
    #             if n - ind - 1 < window and ind < window:
    #                 arr = t
    #             elif n - ind - 1 < window:
    #                 arr = [t[i] for i in range(ind - window, n)]
    #             elif ind < window:
    #                 arr = [t[i] for i in range(0, ind + window + 1)]
    #             else:
    #                 arr = [t[i] for i in range(ind - window, ind + window + 1)]
    #             inputs_window.append(arr)
    #         else:
    #             inputs_window.append(t)
    #     return inputs_window