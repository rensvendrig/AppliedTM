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


# stem
def stem(dev):
    stemmer = SnowballStemmer('english')
    dev['stem'] = pd.Series()
    dev['stem'] = dev['token'].apply(lambda x: stemmer.stem(x))
    return dev


# lemmatize
def lemmatize(dev):
    wordnet_lemmatizer = WordNetLemmatizer()
    dev['lemma'] = pd.Series()
    dev['lemma'] = dev['token'].apply(lambda x: wordnet_lemmatizer.lemmatize(x))
    return dev


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
    return dev


# vector representation
def load_semantic_model(filepath):
    """Function to get semantic model"""

    # this will create an embedding file that can be load by keyedvectors
    # you only need to do it once if u have to use the model multiple times in a task
    glove2word2vec(glove_input_file=filepath, word2vec_output_file="../model/gensim_glove_vectors.txt")

    word_embedding_model = KeyedVectors.load_word2vec_format("../model/gensim_glove_vectors.txt", binary=False)

    return word_embedding_model


def get_vector_representation(dev, frequency_threshold, modelword_index, num_features, WORD_EMBEDDING_MODEL):
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
    featureVectors = []

    for token in tokens:
        if token in modelword_index:
            featureVec = np.add(featureVec,
                                WORD_EMBEDDING_MODEL[token] / np.linalg.norm(WORD_EMBEDDING_MODEL[token]))

            known_words.append(token)
        else:
            unknown_words.append(token)
            featureVec = np.average(featureVectors)

        featureVectors.append(featureVec)
        nwords = nwords + 1

    #featureVec = np.divide(featureVec, nwords)

    # 3. average feature vector
    counter = 0

    devFeatureVecs = np.zeros((len(list_tokens), num_features), dtype="float32")

    for token in tokens:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(token)))

        devFeatureVecs[counter] = featureVectors[counter]
        counter = counter + 1
    print(unknown_words)
    devFeatureVecs = np.nan_to_num(devFeatureVecs)
    return devFeatureVecs


def add_vectors(dev, frequency_threshold, modelword_index, num_features, WORD_EMBEDDING_MODEL):
    vectors = get_vector_representation(dev, frequency_threshold, modelword_index, num_features, WORD_EMBEDDING_MODEL)
    dev["vector"] = pd.Series()
    dev["vector"] = vectors
    return dev



def get_pos(dev):
    dev_list = dev["token"].tolist()
    pos_tags_dev = pos_tag([i for i in dev_list if i])
    words, tags = zip(*pos_tags_dev)
    pos_list = tags
    dev["pos"] = pos_list
    return dev

def get_in_NegExpList(dev):
    # true/false basic negations
    NegExpList = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non']
    dev['in_NegExpList'] = dev['token'].apply(lambda x: x.lower() in NegExpList)
    return dev

def main():
    # load dataset
    path = "SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt"
    dev = pd.read_csv("../data/preprocessed_"+path, sep="\t")

    # run functions
    dev = get_pos(dev)
    dev = stem(dev)
    dev = has_negation_affix(dev)
    dev = get_in_NegExpList(dev)

    WORD_EMBEDDING_MODEL = load_semantic_model("../model/glove.6B.100d.txt")
    dev = add_vectors(dev, 5, set(WORD_EMBEDDING_MODEL.wv.index2word), 100, WORD_EMBEDDING_MODEL)

    # save dataset
    dev.to_csv("../data/featured_"+path, sep="\t",
               header=['chapter', 'sentence_id', 'token_id', 'token', 'target', "pos", "stem",
                       "has_negation_affix", "in_NegExpList", "vector"], index=False)


if __name__ == '__main__':
    main()