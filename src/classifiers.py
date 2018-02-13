from TfidfExtractor import *
import pandas as pd
import time
import itertools
import numpy as np
from time import time
import matplotlib.pyplot as plt

from sklearn import metrics, svm
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier, \
    LogisticRegression


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
    print("Stats :")
    print(metrics.classification_report(dataframe_test.code, pred))
    print()
    print("Confusion Matrix: ")
    cm = metrics.confusion_matrix(dataframe_test.code, pred)
    plotConfusionMatrix(name, confusion_matrix=cm)
    return Result(name, score, train_time, test_time)


def plotConfusionMatrix(clf_name, confusion_matrix):
    plt.figure()
    classes = ["FAKE", "REAL"]
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix : " + clf_name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


dataframe_train = pd.read_csv("../dataset/train_bis.csv")
dataframe_test = pd.read_csv("../dataset/test_OK.csv")

extractor = TfidfExtractor(ngram=1)

train_features = extractor.extract_train(dataframe_train)
test_features = extractor.extract_test(dataframe_test)

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
        # (KNeighborsClassifier(n_neighbors=10), "kNN"),
        # (svm.SVC(kernel='linear', C=0.01), 'SVM'),
        (MultinomialNB(), 'Naive Bayes'),
        (LogisticRegression(), 'Logistic regression'),
        (SGDClassifier(alpha=.0001, max_iter=50, penalty="elasticnet"), 'SGD Elastic Net'),
        (NearestCentroid(), 'Nearest centroid')
):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, name))

indices = np.arange(len(results))

clf_names = [res.clf for res in results]
score = [res.score for res in results]
training_time = [res.training_time for res in results]
test_time = [res.testing_time for res in results]

# Ranking

results.sort(key=lambda x: x.score, reverse=True)
d = {"clf": clf_names, "scores": score, "training time": training_time, "testing time": test_time}

ranking = pd.DataFrame(data=d)
ranking = ranking.sort_values(by="scores", ascending=False)
new_index = [i for i in range(1, len(ranking.index) + 1)]
ranking.index = new_index
print("RANKING :")
print(ranking)
plt.show()
