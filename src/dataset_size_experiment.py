import pandas as pd
from scipy import sparse
from extractFeatures import *
from sklearn.linear_model import RidgeClassifier
import numpy as np
from time import time
from tqdm import *
from sklearn import metrics
dataframe_train = pd.read_csv("../dataset/train_80.csv")

test = pd.read_csv("../dataset/test_20.csv")
dataframe_train = dataframe_train.reindex(np.random.permutation(dataframe_train.index))

size = dataframe_train.shape[0]

results = []

class Result:
    def __init__(self, clf, score, training_time, testing_time):
        self.clf = clf
        self.score = score
        self.training_time = training_time
        self.testing_time = testing_time



def benchmark(clf, name, train, test):
    t0 = time()
    clf.fit(train_features, train.code)
    train_time = time() - t0

    t0 = time()
    pred = clf.predict(test_features)
    test_time = time() - t0

    score = metrics.accuracy_score(test.code, pred)

    # print("Stats :")
    # print(metrics.classification_report(test.code, pred))
    # print()

    return Result(name, score, train_time, test_time)

for percent in tqdm([0.05, 1, 5, 15, 25, 50, 75, 100]):

    to_keep = int((percent/100) * size)
    data = dataframe_train.iloc[0:to_keep]

    print(data.shape[0])

    extractor = PunctuationExtractor()
    punctuation_train = extractFeatureWithVectorizer(extractor, 'punctuation_tfidf', data, train=True)
    punctuation_test = extractFeatureWithVectorizer(extractor, 'punctuation_tfidf', test)

    extractor = PronounsExtractor()
    pronouns_train = extractFeatureWithVectorizer(extractor, 'pronouns_tfidf', data, train=True)
    pronouns_test = extractFeatureWithVectorizer(extractor, 'pronouns_tfidf', test)

    extractor = TfidfExtractor(ngram=1)
    tfidf_train = extractFeatureWithVectorizer(extractor, 'tfidf_' , data, train=True)
    tfidf_test = extractFeatureWithVectorizer(extractor, 'tfidf_', test)

    train_features = sparse.hstack([tfidf_train, punctuation_train, pronouns_train])
    test_features = sparse.hstack([tfidf_test, punctuation_test, pronouns_test])

    clf = RidgeClassifier(tol=1e-2, solver="sag")

    results.append((benchmark(clf, "Ridge Classifier", data, test), percent))

for result, percent in results:
    print(result.score)
