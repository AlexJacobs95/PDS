import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from TfidfExtractor import TfidfExtractor


class Predictor:
    def __init__(self):
        print("Initializing predictor")
        print("Reading data")
        dataframe_train = pd.read_csv("../dataset/train_80.csv")
        tfidf_train = sparse.load_npz('../features/tfidf_train_features.npz')

        print("Training classifier")
        self.classifier = SGDClassifier(alpha=.0001, max_iter=50, penalty="elasticnet")
        self.classifier.fit(tfidf_train, dataframe_train.code)

        print("Loading and training TfidfVectorizer needed to feature extraction later on")
        max_features = 500
        self.tfidfVectorizer = TfidfExtractor(ngram=1, max_features=max_features)
        self.tfidfVectorizer.extract_train(dataframe_train)

    def predict(self, article):
        features = self.tfidfVectorizer.extract_train(article)
        return self.classifier.predict(features)
