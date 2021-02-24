import pandas as pd
from models import Input, PolarityOutput
import numpy as np

# test each aspect:
def load_polarity_data_du_lich(path, label):
    """

    :param path:
    :return:
    :rtype: list of models.Input
    """
    inputs = []
    outputs = []
    df = pd.read_csv(path)
    aspects = 'aspect{}'.format(label+1)
    for _, r in df.iterrows():
        if r[aspects] == -1 or r[aspects] == 1:
            t = r['text'].strip()
            inputs.append(Input(t))
            scores = r[aspects]
            labels = label
            outputs.append(PolarityOutput(labels, aspects, scores))
    return inputs, outputs
# def load_polarity_data_du_lich(path):
#     """
#
#     :param path:
#     :return:
#     :rtype: list of models.Input
#     """
#     inputs = []
#     outputs = []
#     df = pd.read_csv(path)
#
#     for _, r in df.iterrows():
#         t = r['text'].strip()
#
#         inputs.append(Input(t))
#         labels = list(range(1))
#         aspects = [0 if r['aspect{}'.format(i)] == 0 else 1 for i in range(1, 6)]
#         scores = [r['aspect{}'.format(i)] for i in range(1,6)]
#         # Aspects = ['aspect{}'.format(i) for i in range(1, 6) if
#         #            r['aspect{}'.format(i)] == 1 or r['aspect{}'.format(i)] == -1]
#         # scores = [r[i] for i in Aspects]
#         outputs.append(PolarityOutput(labels, aspects, scores))
#     # inputs = [np.array([Input(r['text']) for _, r in df.iterrows() if outputs[i].aspects == 1])
#     #           for i in range(5)]
#     print(inputs)
#     return inputs, outputs

# print(outputs[0].labels)
# print(outputs[0].aspects)
# print(outputs[0].scores)





# for _, r in df.iterrows():
#     t = r['text'].strip()
#     labels = [list(range(5))]
#     for i in range(5):
#         aspects = 'aspect{}'.format(i)
#         if r[aspects] == -1 or r[aspects] == 1:
#             inputs.append(Input(t))
#             scores = r[aspects]
#             outputs.append(PolarityOutput(labels[i], aspects, scores))

# print(outputs[0].labels)
# print(outputs[0].aspects)
# print(outputs[0].scores)



# end test

# def load_polarity_data_du_lich(path):
#     """
#
#     :param path:
#     :return:
#     :rtype: list of models.Input
#     """
#     inputs = []
#     outputs = []
#     df = pd.read_csv(path)
#     for _, r in df.iterrows():
#         t = r['text'].strip()
#         inputs.append(Input(t))
#         labels = list(range(1))
#         aspects = [0 if r['aspect{}'.format(i)] == 0 else 1 for i in range(1, 6)]
#         # scores = [r['aspect{}'.format(i)] for i in range(1,6)]
#         Aspects = ['aspect{}'.format(i) for i in range(1, 6) if r['aspect{}'.format(i)]==1 or r['aspect{}'.format(i)]==-1]
#         scores = [r[i] for i in Aspects]
#         outputs.append(PolarityOutput(labels, aspects, scores))
#     print(outputs[0].labels)
#     print(outputs[0].aspects)
#     print(outputs[0].scores)
#
#     return inputs, outputs


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
