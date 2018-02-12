import pandas as pd
import numpy as np


class SentimentExtractor:
    """ Count the number of good and bad words in each article, based on an annotated dataset
        Returns a csv file with the positive, negative and neutral frequencies per article """

    def __init__(self, file):
        self.positive = []
        self.negative = []
        self.neutral = []
        self.load_words(file)

    def load_words(self, file):
        for row in file.iterrows():
            temp = row[1]["priorpolarity"].split("=")
            if len(temp) > 1:
                if temp[1] == "positive":
                    self.positive.append(row[1]["word"].split("=")[1])
                elif temp[1] == "negative":
                    self.negative.append(row[1]["word"].split("=")[1])
                elif temp[1] == "neutral":
                    self.neutral.append(row[1]["word"].split("=")[1])

    def count(self, data):
        result = []
        pos_counter = 0
        neg_counter = 0
        neutral_counter = 0

        temp = data.split()
        for word in temp:
            if word in self.positive:
                pos_counter += 1
            elif word in self.negative:
                neg_counter += 1
            elif word in self.neutral:
                neutral_counter += 1

        result.append([pos_counter / len(temp), neg_counter / len(temp), neutral_counter / len(temp)])
        return result

    def words_classifier(self, data):
        result = []
        for article in data.text:
            result.append(self.count(article))

        return np.vstack(result)


if __name__ == '__main__':
    data = pd.read_csv("../resources/emotion.csv")
    data_article = pd.read_csv("../dataset/test_OK.csv")
    s = SentimentExtractor(data)
    results = s.words_classifier(data_article)

    from utils import saveMatrixAsCSV
    saveMatrixAsCSV(results, columnNames=["positive", "negative", "neutral"], filename="sentiment_result_features.csv")
