import pandas as pd
import time
import itertools
import numpy as np
from time import time
import itertools
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn import metrics, svm
from sklearn import feature_selection
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


def showRanking(results):
    clf_names = [res.clf for res in results]
    score = [res.score for res in results]
    training_time = [res.training_time for res in results]
    test_time = [res.testing_time for res in results]

    results.sort(key=lambda x: x.score, reverse=True)
    d = {"clf": clf_names, "scores": score, "training time": training_time, "testing time": test_time}

    ranking = pd.DataFrame(data=d)
    ranking = ranking.sort_values(by="scores", ascending=False)
    new_index = [i for i in range(1, len(ranking.index) + 1)]
    ranking.index = new_index
    print("RANKING :")
    print(ranking)


def benchmark(clf, name):
    #print('_' * 80)
    #print("Training: ")
    #print(clf)
    t0 = time()
    clf.fit(train_features, dataframe_train.code)
    train_time = time() - t0
    #print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(test_features)
    test_time = time() - t0
    #print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(dataframe_test.code, pred)
    #print("Stats :")
    #print(metrics.classification_report(dataframe_test.code, pred))
    #print()

    cm = metrics.confusion_matrix(dataframe_test.code, pred)
    # plotConfusionMatrix(name, confusion_matrix=cm)

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


if __name__ == '__main__':

    all_features = []

    dataframe_train = pd.read_csv("../dataset/train_80.csv")
    dataframe_test = pd.read_csv("../dataset/test_20.csv")

    tfidf_train = sparse.load_npz('../features/tfidf_train_features.npz')
    tfidf_test = sparse.load_npz('../features/tfidf_test_features.npz')
    all_features.append([tfidf_train, tfidf_test, "tfidf"])

    punctuation_train = sparse.load_npz('../features/punctuations_train_features.npz')
    punctuation_test = sparse.load_npz('../features/punctuations_test_features.npz')
    all_features.append([punctuation_train, punctuation_test, "punctuation"])

    pronouns_train = sparse.load_npz('../features/pronouns_train_features.npz')
    pronouns_test = sparse.load_npz('../features/pronouns_test_features.npz')
    all_features.append([pronouns_train, pronouns_test, "pronouns"])

    text_counts_train = sparse.load_npz('../features/text_count_train_features.npz')
    text_counts_test = sparse.load_npz('../features/text_count_test_features.npz')
    all_features.append([text_counts_train, text_counts_test, "text_count"])

    # readability_train = sparse.load_npz('../features/readablity_train_features.npz')
    # readability_test = sparse.load_npz('../features/readablity_test_features.npz')

    sentiment_train = sparse.load_npz('../features/sentiment_train_features.npz')
    sentiment_test = sparse.load_npz('../features/sentiment_test_features.npz')
    all_features.append([sentiment_train, sentiment_test, "sentiment"])

    best_res = []

    selector = feature_selection.VarianceThreshold()

    for size_of_combinations in range(1, len(all_features) + 1):

        features_combinations = itertools.combinations(all_features, size_of_combinations)

        for combination in features_combinations:
            print("features : ", [feature[2] for feature in combination])

            train_features = sparse.hstack([feature[0] for feature in combination])
            test_features = sparse.hstack([feature[1] for feature in combination])


            # train_features = sparse.load_npz('../features/tfidf_train_features.npz')
            # test_features = sparse.load_npz('../features/tfidf_test_features.npz')


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
                #print('=' * 80)
                #print(name)
                results.append(benchmark(clf, name))
            # showRanking(results)
            # plt.show()

            results.sort(key=lambda x: x.score, reverse=True)

            features_used = [feature[2] for feature in combination]

            print("best score = " + str(results[0].score) + "using " + str(results[0].clf) )

            best_res.append([results[0].score, results[0].clf, [feature for feature in features_used]])

    best_res.sort(key=lambda x: x[0], reverse=True)
    for elem in best_res:
        print(elem)