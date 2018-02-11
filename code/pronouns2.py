import pandas as pd
import numpy as np
import spacy
import time
import string


class PronounExtractor:
    """ Does a part-of-speech tagging to extract all pronouns
    Returns a n x 7 (for the 7 pronouns) matrix of values (n rows, 7 columns)
    each line contains the relative frequency of each pronouns
          """

    def __init__(self):
        self.nlp = spacy.load('en')
        self.pronoun_list = ["i", "you", "she", "he", "it", "we", "they"]

    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def extract(self, article):

        res = [0 for _ in range(len(self.pronoun_list))]
        article = self.remove_punctuation(article)
        size = len(article)
        doc = self.nlp(article)

        for token in doc:
            if token.tag_ == "PRP":
                if token.text.lower() in self.pronoun_list:
                    res[self.pronoun_list.index(token.text.lower())] += 1

        res = [val / size for val in res]

        return res

    def transform(self, data):
        results = []
        i = 1
        for article in data:
            print(i)
            results.append(self.extract(article))
            i += 1
        return np.vstack(results)


if __name__ == '__main__':
    dataframe_test = pd.read_csv("../dataset/balancedtest_bis.csv")
    extractor = PronounExtractor()
    start = time.time()
    results = extractor.transform(dataframe_test.text)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)
    df = pd.DataFrame(results, columns=["i", "you", "she", "he", "it", "we", "they"])
    df.to_csv("pronouns_feature.csv")
