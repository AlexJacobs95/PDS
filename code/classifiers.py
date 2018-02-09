from TfidfExtractor import *
import pandas as pd
import time
import numpy as np
from time import time
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier, \
    LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn import metrics


class Result:
    def __init__(self, clf, score, training_time, testing_time):
        self.clf = clf
        self.score = score
        self.training_time = training_time
        self.testing_time = testing_time


def benchmark(clf, name):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(train_features, dataframe_train.code)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(test_features)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(dataframe_test.code, pred)
    print("accuracy:   %0.3f" % score)
    print()
    return Result(name, score, train_time, test_time)


dataframe_train = pd.read_csv("../dataset/train_bis.csv")
dataframe_test = pd.read_csv("../dataset/balancedtest_bis.csv")

extractor = TfidfExtractor(ngram=1)

train_features = extractor.extract_train(dataframe_train)
test_features = extractor.extract_test(dataframe_test)


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (LogisticRegression(), 'Logistic regression'),
        (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), 'SGD Elastic Net'),
        (NearestCentroid(), 'Nearest centroid')):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, name))

# Plots

indices = np.arange(len(results))

clf_names = [res.clf for res in results]
score = [res.score for res in results]
training_time = [res.training_time for res in results]
test_time = [res.testing_time for res in results]

training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

# Ranking

results.sort(key=lambda x: x.score, reverse=True)
d = {"clf": clf_names, "scores": score}

ranking = pd.DataFrame(data=d)
ranking = ranking.sort_values(by="scores", ascending=False)
new_index = [i for i in range(1, len(ranking.index) + 1)]
ranking.index = new_index
print("RANKING :")
print(ranking)
