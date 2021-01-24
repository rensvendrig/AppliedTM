import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
from sklearn import svm
from sklearn.metrics import classification_report
import time



train = pd.read_csv("../data/featured_SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt", sep="\t", header = 0)
dev = pd.read_csv("../data/featured_SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt", sep="\t", header = 0)
#1. specify features and target
train_features = train.drop(["chapter","sentence_id", "token_id","target"], axis=1)
dev_features = dev.drop(["chapter","sentence_id", "token_id","target"], axis=1)

train_features_dict = train_features.to_dict("records")
dev_features_dict = dev_features.to_dict("records")
y_train=[x.upper() for x in train["target"].values]
y_test=[x.upper() for x in dev["target"].values]

#2. vectorize features
vec = DictVectorizer(sparse=False)
x_train = np.nan_to_num(vec.fit_transform(train_features_dict))
x_test = np.nan_to_num(vec.transform(dev_features_dict))
#4. generate model

tic = time.perf_counter()
clf = svm.LinearSVC(verbose=2) # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

toc = time.perf_counter()
print(f"Trained in {toc - tic:0.4f} seconds")
#5. predict
y_pred = clf.predict(x_test)

#6. save the classifier to disk
filename_classifier = '../model/svm_linear_clf_embeddings.sav'
pickle.dump(clf, open(filename_classifier, 'wb'))

#7. Classifiaction report
print(classification_report(y_test, y_pred))