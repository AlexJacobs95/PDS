import pandas as pd
import numpy as np
import spacy

class PronounExtractor:
    """ Does a part-of-speech tagging to extract all pronouns
         Returns a n x 7 (for the 7 pronouns) matrix of values (n rows, 7 columns)  """

    def __init__(self):
        self.nlp = spacy.load('en')
        self.pronoun_list = ["i", "you", "she", "he", "it", "we", "they"]

    def extract(self, article):

        res = [0 for _ in range(len(self.pronoun_list))]
        doc = self.nlp(article)
        for token in doc:
                if token.tag_ == "PRP":
                    if token.text.lower() in self.pronoun_list:
                        res[self.pronoun_list.index(token.text.lower())] += 1
        return res

    def transform(self, data):
        results = []
        i = 1
        for article in data:
            print(i)
            results.append(self.extract(article))
            i+=1
        return np.vstack(results)


if __name__ == '__main__':
    dataframe_test = pd.read_csv("../dataset/train_bis.csv")
    extractor = PronounExtractor()
    results = extractor.transform(dataframe_test.text)
    print(results)
