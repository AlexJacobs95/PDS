import pandas as pd
import numpy as np
import spacy
import time
import string

from tqdm import tqdm


def makeDict(pronounList):
    return {pronoun: index for (index, pronoun) in enumerate(pronounList)}


class PronounExtractor:
    """
    Does a part-of-speech tagging to extract all categories of pronouns treated by spacy
    Returns a n x len(pronoun list) matrix of values for each category
    Each line contains the relative frequency of each pronoun
    Convert the results in csv file.
    """

    def __init__(self):
        self.nlp = spacy.load('en')
        self.personal_pronoun_dict = makeDict(["i", "you", "she", "he", "it", "we", "they"])
        self.possessive_pronoun_dict = makeDict(["mine", "yours", "his", "hers", "ours", "theirs"])
        self.wh_personal_pronoun_dict = makeDict(["what", "who", "whom"])
        self.wh_possessive_pronoun_dict = makeDict(["whose", "whosever"])  # {whose: 0, whosever: 1}

    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def extract(self, doc, size, p_dict, p_tag):
        res = [0] * len(p_dict.items())  # Liste de zeros avec la taille du dictionnaire
        # print(len(res))
        for token in doc:
            if token.tag_ == p_tag:
                lowerText = token.text.lower()
                if lowerText in p_dict:
                    res[p_dict[lowerText]] += 1

        res = [val / size for val in res]

        return res

    def transform(self, data):
        personal_results = []
        possessive_results = []
        wh_personal_results = []
        wh_possessive_results = []

        for article in tqdm(data):
            article = self.remove_punctuation(article)
            doc = self.nlp(article)
            size = len(article)
            personal_results.append(self.extract(doc, size, self.personal_pronoun_dict, "PRP"))
            possessive_results.append(self.extract(doc, size, self.possessive_pronoun_dict, "PRP$"))
            wh_personal_results.append(self.extract(doc, size, self.wh_personal_pronoun_dict, "WP"))
            wh_possessive_results.append(self.extract(doc, size, self.wh_possessive_pronoun_dict, "WP$"))

        return np.vstack(personal_results), np.vstack(possessive_results), np.vstack(wh_personal_results), np.vstack(
            wh_possessive_results)


if __name__ == '__main__':
    dataframe_test = pd.read_csv("../dataset/test_OK.csv")
    extractor = PronounExtractor()
    start = time.time()

    r1, r2, r3, r4 = extractor.transform(dataframe_test.text)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    from utils import saveMatrixAsCSV

    saveMatrixAsCSV(matrix=r1, columnNames=["i", "you", "she", "he", "it", "we", "they"],
                    filename="personal_pronouns_feature.csv")
    saveMatrixAsCSV(matrix=r2, columnNames=["mine", "yours", "his", "hers", "ours", "theirs"],
                    filename="possessive_pronouns_feature.csv")
    saveMatrixAsCSV(matrix=r3, columnNames=["what", "who", "whom"], filename="wh_personal_pronouns_feature.csv")
    saveMatrixAsCSV(matrix=r4, columnNames=["whose", "whosever"], filename="wh_possessive_pronouns_feature.csv")
