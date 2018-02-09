from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time


class TfidfExtractor:

    @staticmethod
    def extract(filename, ngram):
        data = pd.read_csv(filename, names = ["code", "text"])
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1, ngram))
        print("Extracting Tfidf...")
        t0 = time.time()
        features = vectorizer.fit_transform(data.text)
        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

        return features


if __name__ == '__main__':
    # test 1-gram
    res = TfidfExtractor.extract("../../CLASSIFIEROFDOOM/fakenewsfiles/fakenewsfiles/train.csv", 1)