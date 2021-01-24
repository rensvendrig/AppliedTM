import pandas as pd
import numpy as np
import enchant

from nltk import pos_tag
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

pd.options.display.width = 0

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from collections import Counter

# load dataset
path = "../data/preprocessed_SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt"
dev = pd.read_csv(path, sep="\t")

# pos
dev_list = dev["token"].tolist()
pos_tags_dev = pos_tag([i for i in dev_list if i])
words, tags = zip(*pos_tags_dev)
pos_list = tags
dev["pos"] = pos_list

# true/false basic negations
NegExpList = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non']
dev['in_NegExpList'] = dev['token'].apply(lambda x: x.lower() in NegExpList)


# stem
def stem(dev):
    stemmer = SnowballStemmer('english')
    dev['stem'] = pd.Series()
    dev['stem'] = dev['token'].apply(lambda x: stemmer.stem(x))


# lemmatize
def lemmatize(dev):
    wordnet_lemmatizer = WordNetLemmatizer()
    dev['lemma'] = pd.Series()
    dev['lemma'] = dev['token'].apply(lambda x: wordnet_lemmatizer.lemmatize(x))


# cues affix
def has_negation_affix(dev):
    prefixes = ['dis', 'un', 'ir', 'im', 'in']
    suffixes = ['less']
    has_negation_affix_list = []
    d = enchant.Dict("en_US")
    # check all tokens starting with a prefix of 2 letters
    for token in dev.token:
        if len(token) > 4:
            if str(token).startswith(tuple(prefixes[1:])):
                has_negation_affix_list.append(d.check(token[2:]))
            # check all tokens starting with a prefix of 3 letters
            elif str(token).startswith(prefixes[0]):
                has_negation_affix_list.append(d.check(token[3:]))
            # check all tokens ending with a suffix of 3 letters
            elif str(token).endswith(suffixes[0]):
                has_negation_affix_list.append(d.check(token[:-4]))
            else:
                has_negation_affix_list.append(False)
        else:
            has_negation_affix_list.append(False)

    dev['has_negation_affix'] = has_negation_affix_list


# vector representation
def load_semantic_model(filepath):
    """Function to get semantic model"""

    # this will create an embedding file that can be load by keyedvectors
    # you only need to do it once if u have to use the model multiple times in a task
    glove2word2vec(glove_input_file=filepath, word2vec_output_file="../model/gensim_glove_vectors.txt")

    word_embedding_model = KeyedVectors.load_word2vec_format("../model/gensim_glove_vectors.txt", binary=False)

    return word_embedding_model


def get_vector_representation(dev):
    tokens = dev["token"]
    list_tokens = list(tokens)

    # 1.
    kw_counter = Counter(list_tokens)

    frequent_keywords = []

    for word, count in kw_counter.items():
        if count > frequency_threshold:
            frequent_keywords.append(word)

    # 2.
    featureVec = np.zeros(num_features, dtype="float32")

    nwords = 0

    known_words = []
    unknown_words = []

    for token in tokens:
        if token in frequent_keywords:
            if token in modelword_index:
                featureVec = np.add(featureVec,
                                    WORD_EMBEDDING_MODEL[token] / np.linalg.norm(WORD_EMBEDDING_MODEL[token]))
                known_words.append(token)
                nwords = nwords + 1
            else:
                if token in modelword_index:
                    featureVec = np.add(featureVec,
                                        WORD_EMBEDDING_MODEL[token] / np.linalg.norm(WORD_EMBEDDING_MODEL[token]))
                    known_words.append(token)
                    nwords = nwords + 1
                else:
                    unknown_words.append(token)

    featureVec = np.divide(featureVec, nwords)

    # 3. average feature vector
    counter = 0

    devFeatureVecs = np.zeros((len(list_tokens), num_features), dtype="float32")

    for token in tokens:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(token)))

        devFeatureVecs[counter] = featureVec
        counter = counter + 1

    devFeatureVecs = np.nan_to_num(devFeatureVecs)
    return devFeatureVecs


def add_vectors(dev):
    vectors = get_vector_representation(dev)
    dev["vector"] = pd.Series()
    dev["vector"] = vectors


# define variables
model_path = "../model/glove.6B.100d.txt"
treshold = 5
WORD_EMBEDDING_MODEL = load_semantic_model(model_path)
modelword_index = set(WORD_EMBEDDING_MODEL.wv.index2word)

frequency_threshold = 5
num_features = 100
nwords = 5

# run functions
stem(dev)
has_negation_affix(dev)
add_vectors(dev)

# save dataset
dev.to_csv("../data/featured_SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt", sep="\t",
           header=['chapter', 'sentence_id', 'token_id', 'token', 'target', "pos", "in_NegExpList", "stem",
                   "has_negation_affix", "vector"], index=False)
