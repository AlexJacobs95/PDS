import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from utils import saveMatrixAsCSV


class SentimentExtractor:
    def extract(self, data):
        output = []
        for news in data.text:
            news_count = 0
            blob = TextBlob(news, analyzer=NaiveBayesAnalyzer())
            output.append([blob.sentiment.p_pos, blob.sentiment.p_neg])

        return np.vstack(output)


if __name__ == '__main__':
    data = pd.read_csv("../dataset/test_OK.csv")
    s_extractor = SentimentExtractor()
    results = s_extractor.extract(data)

    saveMatrixAsCSV(results, columnNames=["positive", "negative"], filename='sentiment_features.csv')
