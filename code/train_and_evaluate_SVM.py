import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import time


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

def SVM(dev, train):
    # 1. specify features and target
    train_features = train.drop(["chapter", "sentence_id", "token_id", "target"], axis=1)
    dev_features = dev.drop(["chapter", "sentence_id", "token_id", "target"], axis=1)

    train_features_dict = train_features.to_dict("records")
    dev_features_dict = dev_features.to_dict("records")
    y_train = [x.upper() for x in train["target"].values]
    y_test = [x.upper() for x in dev["target"].values]

    # 2. vectorize features
    vec = DictVectorizer(sparse=False)
    x_train = np.nan_to_num(vec.fit_transform(train_features_dict))
    x_test = np.nan_to_num(vec.transform(dev_features_dict))
    # 4. generate model

    tic = time.perf_counter()
    clf = svm.LinearSVC(verbose=2)  # Linear Kernel

    # Train the model using the training sets
    clf.fit(x_train, y_train)

    toc = time.perf_counter()
    print(f"Trained in {toc - tic:0.4f} seconds")

    return clf, x_test, y_test

def main():
    train = pd.read_csv("../data/featured_SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt", sep="\t", header = 0)
    dev = pd.read_csv("../data/featured_SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt", sep="\t", header = 0)

    clf, x_test, y_test = SVM(dev, train)

    # predict
    y_pred = clf.predict(x_test)

    #6. save the classifier to disk
    filename_classifier = '../model/svm_linear_clf_embeddings.sav'
    pickle.dump(clf, open(filename_classifier, 'wb'))

    #7. Classifiaction report
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    error_analysis(dev, y_test, y_pred)

if __name__ == '__main__':
    main()