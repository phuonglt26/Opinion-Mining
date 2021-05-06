import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from transformers import AutoTokenizer, AutoModel

from models import PolarityOutput
from modules.models import Model

#Phobert
class PolarityphobertModel(Model):

    def __init__(self):
        self.NUM_OF_ASPECTS = 6
        self.vocab = []
        self.models = [LogisticRegression() for _ in range(self.NUM_OF_ASPECTS)]

    def _represent(self, inputs, aspectId):
        phobert = AutoModel.from_pretrained("vinai/phobert-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        a = []
        for t in inputs:
            input_ids = torch.tensor([tokenizer.encode(t.text)])
            with torch.no_grad():
                features = phobert(input_ids)
                s = np.array(features[1])
                # print(s[0])
                a.append(s[0])
        return np.array(a)

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
