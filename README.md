# AppliedTM
This repository is created to train and test two negation cue detection classifiers. This project is done via the Applied Text Mining course on the VU. 

annotations: This folder contains negation cue annotations for 10 articles. 

data: This folder contains the data used for training and testing both systems.
The glove.6b.100d.txt file in the model folder is not included since it is too large to upload to github. Find and download it here: https://nlp.stanford.edu/projects/glove/

code: This folder contains the following python scripts:

preprocess.py: This file preprocesses the corpus

feature_extraction.py: This file extracts features from the preprocessed data file.

train_and_evaluate_CRF: This file trains, tests and evaluates the CRF model. 

train_and_evaluate_SVM: This file trains, tests and evaluates the SVM model. 
