import models
import pandas as pd

LABEL_MAP = {
    313: "cấu hình -", # tiki_tech
    312: "cấu hình +", # tiki_tech
    311: "mẫu mã -", # tiki_tech
    310: "mẫu mã +", # tiki_tech
    309: "hiệu năng -", # tiki_tech
    308: "hiệu năng +", # tiki_tech
    307: "dịch vụ -", # tiki_tech
    306: "dịch vụ +", # tiki_tech
    305: "chính hãng -", # tiki_tech
    304: "chính hãng +", # tiki_tech
    303: "phụ kiện -", # tiki_tech
    302: "phụ kiện +", # tiki_tech
    301: "ship -", # tiki_tech
    300: "ship +", # tiki_tech
    299: "giá -", # tiki_tech
    298: "giá +", # tiki_tech
    297: "other", # tiki_tech
    296: "trash", # tiki_tech
    295: "typo", # tiki_tech
    328: "an toàn -", # tiki_mebe
    327: "an toàn +", # tiki_mebe
    326: "dịch vụ -", # tiki_mebe
    325: "dịch vụ +", # tiki_mebe
    324: "chính hãng -", # tiki_mebe
    323: "chính hãng +", # tiki_mebe
    322: "chất lượng -", # tiki_mebe
    321: "chất lượng +", # tiki_mebe
    320: "ship -", # tiki_mebe
    319: "ship +", # tiki_mebe
    318: "giá -", # tiki_mebe
    317: "giá +", # tiki_mebe
    316: "other", # tiki_mebe
    315: "trash", # tiki_mebe
    314: "typo", # tiki_mebe
    347: "cấu hình -", # shopee_tech
    346: "cấu hình +", # shopee_tech
    345: "mẫu mã -", # shopee_tech
    344: "mẫu mã +", # shopee_tech
    343: "hiệu năng -", # shopee_tech
    342: "hiệu năng +", # shopee_tech
    341: "dịch vụ -", # shopee_tech
    340: "dịch vụ +", # shopee_tech
    339: "chính hãng -", # shopee_tech
    338: "chính hãng +", # shopee_tech
    337: "phụ kiện -", # shopee_tech
    336: "phụ kiện +", # shopee_tech
    335: "ship -", # shopee_tech
    334: "ship +", # shopee_tech
    333: "giá -", # shopee_tech
    332: "giá +", # shopee_tech
    331: "other", # shopee_tech
    330: "trash", # shopee_tech
    329: "typo", # shopee_tech
    362: "an toàn -", # shopee_mebe
    361: "an toàn +", # shopee_mebe
    360: "dịch vụ -", # shopee_mebe
    359: "dịch vụ +", # shopee_mebe
    358: "chính hãng -", # shopee_mebe
    357: "chính hãng +", # shopee_mebe
    356: "chất lượng -", # shopee_mebe
    355: "chất lượng +", # shopee_mebe
    354: "ship -", # shopee_mebe
    353: "ship +", # shopee_mebe
    352: "giá -", # shopee_mebe
    351: "giá +", # shopee_mebe
    350: "other", # shopee_mebe
    349: "trash", # shopee_mebe
    348: "typo", # shopee_mebe
}

def load_data(path):
    """

    :param path:
    :return:
    :rtype: list of models.Input
    """
    pass


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


def preprocess(inputs, break_sentence, dup_letter_filter, stopword_filter, acronym_filter, punctuation_remove,
               tokenizer: models.Tokenizer=None,
               stopword=None,
               acronym=None):
    """
    :param tokenizer: pretrain model for tokenizing
    :param dup_letter_filter: remove duplicate letter at the end of word. e.g: hiiiiiiii -> hi
    :param punctuation_remove: remove punctuation like . , ' \ ...
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

    ans = inputs

    if break_sentence:
        ans = tokenizer.tokenizer(ans)

    ans = word_filter(
        inputs=ans,
        use_dup_letter_filter=dup_letter_filter,
        use_acronym=acronym_filter,
        use_stopword=stopword_filter,
        use_punctuation_remove=punctuation_remove,
        acronyms=acronym,
        stopword=stopword
    )

    return ans


def word_filter(inputs, use_dup_letter_filter, use_acronym, use_stopword, use_punctuation_remove,
                acronyms : pd.DataFrame = None,
                stopword : pd.DataFrame = None):
    """
    Multi function filter
    :param inputs: list of models.Input
    :param use_dup_letter_filter: True if remove duplicated last letter. ex: hiiiii -> hi
    :param use_acronym: True, replace acronym with meaning
    :param use_punctuation_remove: True, remove all punctuation, ALL
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
            # replace acronym with corresponding word
            if use_acronym and text in acronyms["acronym"].values:
                text = acronyms[acronyms["acronym"] == text].iloc[0, 1]
            # remove stopword
            if use_stopword and text in stopword.values:
                continue
            if use_punctuation_remove:
                if not 'a' <= text[0] <= 'z':
                    continue

            ans.append(text)
        outputs.append(models.Input(" ".join(ans)))
    return outputs


"""import pandas as pd
from models import *

inputs = []
inputs.append(Input("Hàng hóa hơi cũ. Đồ ăn ngon nhưng rất bẩn."))
inputs.append(Input("Mặt nạ nhìn đẹp, nhưng giá cả đắt đỏ."))

acronyms = pd.read_csv("E:\\MachineLearning\\Study\\ml-dev-log\\Document\\acronym.txt", sep=',', header=None, names=["acronym","meaning"])
acronyms.head()

stopword = pd.read_csv("E:\\MachineLearning\\Study\\ml-dev-log\\Document\\vietnamese_stopwords.txt", sep=',', header=None, names=["word"])
stopword.head()

inputs = []
inputs.append(Input("Hàng hóa hơi cũ. Đồ ăn ngon nhưng rất bẩn. kjsuegjsjergns. hiiiii"))
inputs.append(Input("Mặt nạ nhìn đẹp, nhưng giá cả đắt đỏ. Còn sp đt dùng rất nhanh hỏng."))

tokenizer = models.PivyTokenizer()

output = preprocess(
    inputs,
    True,
    True,
    True,
    True,
    True,
    tokenizer,
    stopword,
    acronyms
)

for sentence in output:
    print(sentence.text)"""



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
