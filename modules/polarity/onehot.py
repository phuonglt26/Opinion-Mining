import pickle

import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from transformers import AutoModel, AutoTokenizer

from models import PolarityOutput
from modules.models import Model


# one-hot
class PolarityonehotModel(Model):
    def __init__(self):
        self.NUM_OF_ASPECTS = 6
        self.vocab = []
        labelVocab = ["giá", "dịch_vụ", "an_toàn", "chất_lượng", "ship", "chính_hãng"]
        for label in labelVocab:
            _vocab = []
            with open('data/vocab/mebe_tiki/label_{}_mebe_tiki.txt'.format(label), encoding="utf-8") as f:
                for l in f:
                    l = l.split(',')
                    _vocab.append(l)
            self.vocab.append(_vocab)
        self.models = [KNeighborsClassifier() for _ in range(self.NUM_OF_ASPECTS)]
        # RandomForestClassifier
        # LogisticRegression
        # MultinomialNB
        # KNeighborsClassifier
        # DecisionTreeClassifier
        # SVC

    def _represent(self, inputs, aspectId):
        features = []
        for ip in inputs:
            _features = [1 if v[0] in ip.text else 0 for v in
                         self.vocab[aspectId]]
            features.append(_features)
        # print(features)

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
            # if g.scores == p.scores == -1:
            #     tp += 1
            # elif g.scores == -1:
            #     fn += 1
            # elif p.scores == -1:
            #     fp += 1
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
