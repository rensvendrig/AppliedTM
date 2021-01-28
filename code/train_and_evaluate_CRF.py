import pandas as pd
import pycrfsuite
from sklearn.metrics import confusion_matrix, classification_report
import time

def featurize(sentence_frame, current_idx):
    current_token = sentence_frame.iloc[current_idx]
    vector = current_token['vector']
    has_negation_affix = current_token['has_negation_affix']
    stem = current_token['stem']
    in_NegExpList = current_token['in_NegExpList']
    pos = current_token['pos']

    # Shared features across tokens
    features = {
            'vector': vector,
            'has_negation_affix':has_negation_affix,
            'stem':stem,
            'in_NegExpList': in_NegExpList,
            'postag': pos
    }

    return features

def featurize_sentence(sentence_frame):
    labels = list(sentence_frame['target'].values)
    features = [featurize(sentence_frame, i) for i in range(len(sentence_frame))]

    return features, labels

def rollup(dataset):
    sequences = []
    labels = []
    offers = dataset.groupby(['chapter','sentence_id'])
    for name, group in offers:
        sqs, lbls = featurize_sentence(group)
        sequences.append(sqs)
        labels.append(lbls)

    return sequences, labels

def error_analysis(test_docs, test_labels, pred_labels):
    numWrong = 0
    i = 0
    wrongTokens = []
    for test_label, pred_label in zip(test_labels, pred_labels):
        if test_label != pred_label:
            print("true value: ",test_label,"\t predicted: ", pred_label, " on line ", i)
            print("line:", test_docs.iloc[i, :])
            print("\n")
            wrongTokens.append(test_docs.loc[i, "token"])
            numWrong += 1
        i += 1
    print(wrongTokens)
    print("total of ", numWrong, " instances incorrectly predicted")
    return True

def main():
    trainFile = "SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt"
    devFile = "SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt"

    train = pd.read_csv("../data/featured_" + trainFile, sep="\t", header = 0)
    dev = pd.read_csv("../data/featured_" + devFile, sep="\t", header=0)

    train_docs, train_labels = rollup(train)
    test_docs, test_labels = rollup(dev)

    tic = time.perf_counter()

    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
    })

    for xseq, yseq in zip(train_docs, train_labels):
        trainer.append(xseq, yseq)

    trainer.train('../model/vuelax-bad.crfsuite')

    toc = time.perf_counter()
    print(f"Trained in {toc - tic:0.4f} seconds")
    crf_tagger = pycrfsuite.Tagger()
    crf_tagger.open('../model/vuelax-bad.crfsuite')

    all_true, all_pred = [], []

    for i in range(len(test_docs)):
        all_true.extend(test_labels[i])
        all_pred.extend(crf_tagger.tag(test_docs[i]))
    print(classification_report(all_true, all_pred))
    print(confusion_matrix(all_true, all_pred))
    error_analysis(dev, all_true, all_pred)


if __name__ == '__main__':
    main()