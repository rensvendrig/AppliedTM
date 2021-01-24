import pandas as pd
import nltk
import string

from nltk.corpus import stopwords

path = "../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt"
dev = pd.read_csv(path, sep="\t", header=None)
dev.columns = ['chapter', 'sentence_id', 'token_id', 'token', 'target']

# lower
dev.token = dev.token.str.lower()

# remove stopwords
stopword_list = stopwords.words("english")

possible_negation_cues = ["not", "no", "never", "nor", "than", "nothing", "never", "cannot"]

new_stopword_list = []

for words in stopword_list:
    if (words not in possible_negation_cues) & ("n't" not in words):
        new_stopword_list.append(words)

for words in new_stopword_list:
    dev.token = dev.token.replace(words, "")

# remove numbers
dev.token = dev.token.str.replace("\d+", "")

# remove punctuation
punctuations = string.punctuation

for punct in punctuations:
    dev.token = dev.token.replace(punct, "")
dev = dev.replace('``', "")
dev = dev.replace("''", "")
dev = dev.replace("--", "")

# remove empty cell rows
new_dev = dev[dev.token != ""]

# save preprocessed dataset
new_dev.to_csv("../data/preprocessed_SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt", sep="\t", header=True,
               index=False)