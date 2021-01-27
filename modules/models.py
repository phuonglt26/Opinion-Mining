

class Model:
    def train(self, inputs, outputs):
        """

        :param inputs:
        :param outputs:
        :return:
        """
        raise NotImplementedError

    def save(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def load(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        :rtype: list of models.Output
        """
        raise NotImplementedError

        
from pyvi import ViTokenizer, ViPosTagger
from vncorenlp import *


# class Document:
#
# class Sentence:
#
# class Token:


class Input:
    def __init__(self, text):
        self.text = text


class AspectOutput:
    def __init__(self, aspects, scores):
        self.aspects = aspects
        self.scores = scores


class PolarityOutput:
    def __init__(self, labels, aspects, scores):
        self.labels = labels
        self.aspects = aspects
        self.scores = scores


class Tokenizer:
    def tokenizer(self, inputs):
        pass


class VnCoreNLPTokenizer(Tokenizer):
    def __init__(self, path):
        self.model = VnCoreNLP(path)

    def tokenizer(self, inputs):
        output = []
        for input in inputs:
            word_paths = self.model.tokenize(input.text)
            word_paths_copy = [y.lower() for x in word_paths for y in x]
            word_paths = word_paths_copy
            output.append(Input(" ".join(word_paths)))
        return output


class PivyTokenizer(Tokenizer):
    def __init__(self):
        pass

    def tokenizer(self, inputs):
        output = []
        for input in inputs:
            word_paths = ViPosTagger.postagging(ViTokenizer.tokenize(input.text))[0]
            word_paths_copy = [x.lower() for x in word_paths]
            word_paths = word_paths_copy
            output.append(Input(" ".join(word_paths)))
        return output
