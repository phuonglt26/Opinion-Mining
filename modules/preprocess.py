import pandas as pd

from models import Input, PolarityOutput


# test each aspect:
def load_polarity_data(path, aspectId):
    """

    :param path:
    :return:
    :rtype: list of models.Input
    """
    inputs = []
    outputs = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        if r['aspect{}'.format(aspectId)] != 0:
            t = r['text'].strip()
            inputs.append(Input(t))
            score = r['aspect{}'.format(aspectId)]
            label = 'aspect{}'.format(aspectId) + (' -' if score == -1 else ' +')
            aspect = 'aspect{}'.format(aspectId)
            outputs.append(PolarityOutput(label, aspect, score))

    return inputs, outputs


def preprocess(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    return inputs


def preprocess_tiki(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    pass


def preprocess_dulich(inputs):
    """

    :param list of models.Input inputs:
    :return:
    :rtype: list of models.Input
    """
    pass
