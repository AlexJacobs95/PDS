import numpy as np
import pandas as pd


class TextCountExtractor:
    """
    Extracts
     - the number of words in each article
     - the number of characters in each article
    If there are n articles, it returns a n x 2 matrix of values (n rows, 2 columns)
    """

    def getArticleInfo(self, article):
        numChars = len(article)
        numWords = len(article.split())
        return [numChars, numWords]

    def transform(self, data):
        results = [self.getArticleInfo(article) for article in data]
        return np.vstack(results)


if __name__ == '__main__':
    # dataframe_train = pd.read_csv("../dataset/train_bis.csv")
    dataframe_test = pd.read_csv("../dataset/balancedtest_bis.csv")

    extractor = TextCountExtractor()
    results = extractor.transform(dataframe_test.text)

    from utils import saveMatrixAsCSV
    saveMatrixAsCSV(results, columnNames=["numChars", "numWords"], filename="text_count_features.csv")
    print(results)
