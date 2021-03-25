import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models import PolarityOutput
from modules.models import Model


class PolarityRFModel(Model):
    def __init__(self):
        self.NUM_OF_ASPECTS = 6
        self.vocab = []
        labelVocab = ["giá", "dịch_vụ", "an_toàn", "chất_lượng", "ship", "chính_hãng"]
        for label in labelVocab:
            _vocab = []
            with open('data/vocab/label_{}_mebe_tiki.txt'.format(label), encoding="utf-8") as f:
                for l in f:
                    l = l.split(',')
                    _vocab.append(l)
            self.vocab.append(_vocab)
        self.models = [RandomForestClassifier() for _ in range(self.NUM_OF_ASPECTS)]

    def _represent(self, inputs, aspectId):
        """

        :param list of models.Input inputs:
        :return:
        """
        features = []
        for ip in inputs:
            _features = [v[1] if v[0] in ip.text else 0 for v in
                         self.vocab[aspectId]]
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

    def save(self, path):
        pass

    def load(self, path):
        pass

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

    def evaluate(self, y_test, y_predicts):
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
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        return tp, fp, fn, p, r, f1
