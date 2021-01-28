import pandas as pd
import string
from nltk.corpus import stopwords

def lower_tokens(dev):
    dev.token = dev.token.str.lower()
    return dev
def remove_stopwords(dev):
    stopword_list = stopwords.words("english")

    possible_negation_cues = ["not", "no", "never", "nor", "than", "nothing", "never", "cannot"]

    new_stopword_list = []

    for words in stopword_list:
        if (words not in possible_negation_cues) & ("n't" not in words):
            new_stopword_list.append(words)

    for words in new_stopword_list:
        dev.token = dev.token.replace(words, "")

    return dev

def remove_numbers(dev):
    dev.token = dev.token.str.replace("\d+", "")
    return dev

def remove_punctuation(dev):
    punctuations = string.punctuation

    for punct in punctuations:
        dev.token = dev.token.replace(punct, "")
    dev = dev.replace('``', "")
    dev = dev.replace("''", "")
    dev = dev.replace("--", "")
    return dev

def main():
    path = "SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt"
    dev = pd.read_csv("../data/"+path, sep="\t", header=None)
    dev.columns = ['chapter', 'sentence_id', 'token_id', 'token', 'target']

    dev = lower_tokens(dev)
    dev = remove_stopwords(dev)
    dev = remove_numbers(dev)
    dev = remove_punctuation(dev)

    # remove empty cell rows
    new_dev = dev[dev.token != ""]

    # save preprocessed dataset
    new_dev.to_csv("../data/preprocessed_" + path, sep="\t", header=True, index=False)

if __name__ == '__main__':
    main()