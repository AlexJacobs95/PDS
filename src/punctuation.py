import pandas as pd
import numpy as np
import spacy
import time
import string


def makeDict(list_):
    return {punct: index for (index, punct) in enumerate(list_)}


class PunctuationExtractor:
    """"""


    def __init__(self):
        self.nlp = spacy.load('en')
        self.punctuation_list = makeDict([punct for punct in string.punctuation])

    def extract(self, doc, size):
        res = [0] * len(self.punctuation_list.items())
        for token in doc:
            if token.text in self.punctuation_list:
                res[self.punctuation_list[token.text]] += 1

        res = [val / size for val in res]

        return res

    def transform(self, data):
        i = 1
        results = []
        for article in data:
            print(i)
            doc = self.nlp(article)
            size = len(article)
            results.append(self.extract(doc, size))
            i += 1

        return np.vstack(results)


if __name__ == '__main__':
    dataframe_test = pd.read_csv("../dataset/test_OK.csv")
    extractor = PunctuationExtractor()
    start = time.time()
    result = extractor.transform(dataframe_test.text)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    df = pd.DataFrame(result, columns=[punct for punct in string.punctuation])
    df.to_csv("../dataset/features/punctuation.csv")
