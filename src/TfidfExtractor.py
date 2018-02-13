from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time


class TfidfExtractor:
    def __init__(self, ngram):
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1, ngram))

    def extract_train(self, data):
        print("Extracting Tfidf...")
        t0 = time.time()
        features = self.vectorizer.fit_transform(data.text)
        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

        return features

    def extract_test(self, data):
        print("Extracting Tfidf...")
        t0 = time.time()
        features = self.vectorizer.transform(data.text)
        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

        return features

    def get_vectorizer(self):
        return self.vectorizer
