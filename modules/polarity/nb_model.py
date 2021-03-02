from sklearn.naive_bayes import MultinomialNB
import numpy as np

from models import PolarityOutput
from modules.models import Model


class HotelPolarityNBModel(Model):
    def __init__(self):
        self.NUM_OF_LABLES = 5
        self.vocab = []
        with open('data/vocab/hotel_vocab.txt', encoding="utf-8") as f:
            for l in f:
                self.vocab.append(l.strip())

        self.models = MultinomialNB()

    def _represent(self, inputs):
        """

        :param list of models.Input inputs:
        :return:
        """
        features = []
        for ip in inputs:
            _features = [1 if v in ip.text else 0 for v in self.vocab]
            features.append(_features)

        return np.array(features)


# test each aspect
    def train(self, inputs, outputs):
        """

        :param list of models.Input inputs:
        :param list of models.AspectOutput outputs:
        :return:
        """
        X = self._represent(inputs)
        ys = [output.scores for output in outputs]
        self.models.fit(X, ys)
# end test


    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.AspectOutput
        """
        X = self._represent(inputs)
        outputs = []
        predicts = self.models.predict(X)
        # labels = 1
        # aspects = 'aspect1'
        # scores = list(predicts)
        # print(scores)
        # outputs.append(PolarityOutput(labels,aspects, scores))
        for ps in predicts:
            labels = 1
            aspects = 'aspect5'
            scores = ps
            outputs.append(PolarityOutput(labels, aspects, scores))
        return outputs

    # def predict(self, inputs):
    #     """
    #
    #     :param inputs:
    #     :return:
    #     :rtype: list of models.AspectOutput
    #     """
    #     X = self._represent(inputs)
    #
    #     outputs = []
    #     predicts = [self.models[i].predict(X) for i in range(self.NUM_OF_ASPECTS)]
    #     for ps in zip(*predicts):
    #         labels = list(range(self.NUM_OF_ASPECTS))
    #         aspects = list(ps)
    #         scores = list(ps)
    #         outputs.append(PolarityOutput(labels,aspects, scores))
    #     return outputs
