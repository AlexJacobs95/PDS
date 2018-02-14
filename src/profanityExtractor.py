import json
import pandas as pd
import time
import numpy as np

from tqdm import tqdm


class ProfanityExtractor:
    """
    Extracts the relative frequency of profanities in each article
    """

    def __init__(self, dictionary):
        self.dictionary = json.load(open(dictionary))

    def getRelFrequencyProfanities(self, article):
        numProfanities = 0
        words = article.split()
        length = len(words)
        for word in words:
            if word in self.dictionary:
                numProfanities += 1

        return numProfanities / length

    def extract(self, data):
        result = []
        print("Extracting profanities...")
        t0 = time.time()
        for article in tqdm(data.text):
            result.append(self.getRelFrequencyProfanities(article))

        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

        return np.transpose(np.matrix(result))


if __name__ == '__main__':
    dataset = pd.read_csv('../dataset/test_OK.csv')
    extractor = ProfanityExtractor('../resources/profanities.json')
    res = extractor.extract(dataset)

    from utils import saveMatrixAsCSV

    #saveMatrixAsCSV(matrix=res, columnNames=["relFrequencyProfanities"], filename="profanities_features.csv")
    print(res)
