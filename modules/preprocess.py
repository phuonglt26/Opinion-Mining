import pandas as pd
import models
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

def load_stopword(path):
    """
    load stopword from path,
    :param path:
    :return: pd.DataFrame
    """
    return pd.read_csv(path, sep=',', header=None, names=["stopword"])


def load_acronym(path):
    """
    load vietnamese acronyms in comments, maybe missing
    :param path:
    :return: pd.DataFrame
    """
    return pd.read_csv(path, sep=',', header=None, names=["acronym", "meaning"])

def preprocess(inputs, stopword=None, acronym=None, break_sentence=True, stopword_filter=True, acronym_filter=True, tokenizer: models.Tokenizer=None):
    """
    :param tokenizer: pretrain model
    :param acronym_filter: using acronym converter or not
    :param acronym: pd.DataFrame acronym: list of acronym
    :param inputs: list of models.Input: inputs
    :param break_sentence: True if break the sentence into words
    :param stopword_filter: True if filter stopword
    :param stopword: pd.DataFrame of stopword
    :return: list of models.Input: output
    """
    # using VnCoreNLP for sentence segmentation
    # default path:
    tokenizer = models.PivyTokenizer()
    ans = inputs

    if break_sentence:
        ans = tokenizer.tokenizer(ans)

    ans = word_filter(ans, acronym_filter, stopword_filter, acronym, stopword)

    return ans


def word_filter(inputs, use_dup_letter_filter, use_len_filter, use_acronym, use_stopword, acronyms : pd.DataFrame = None, stopword : pd.DataFrame = None):
    """
    Multi function filter
    :param inputs: list of models.Input
    :param use_dup_letter_filter: True if remove duplicated last letter. ex: hiiiii -> hi
    :param use_len_filter: True, remove word that longer than 7
    :param use_acronym: True, replace acronym with meaning
    :param use_stopword: True, remove stopword
    :param acronyms: DataFrame of acronyms
    :param stopword: DataFrame of stopwords
    :return:
    """
    outputs = []
    for input in inputs:
        texts = input.text.split(' ')
        ans = []
        for text in texts:
            # remove duplicate last letter
            while use_dup_letter_filter and len(text) > 1 and text[-1] == text[-2]:
                text = text[:-1]
            # remove too long or null word
            if use_len_filter:
                if len(text) > 7 or len(text) < 1:
                    continue
            # replace acronym with corresponding word
            if use_acronym and text in acronyms["acronym"].values:
                text = acronyms[acronyms["acronym"] == text].iloc[0, 1]
            # remove stopword
            if use_stopword and text in stopword.values:
                continue

            ans.append(text)
        outputs.append(models.Input(" ".join(ans)))
    return outputs


def examinate(inputs):
    print("Len: ", len(inputs))
    for input in inputs:
        print(input.text)
    print()

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
