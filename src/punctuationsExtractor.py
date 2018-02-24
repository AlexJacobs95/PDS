import argparse
import time

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class PunctuationStatisticsVectorizer(TfidfVectorizer):

    def __init__(self):
        super(PunctuationStatisticsVectorizer, self).__init__()
        self.nlp = spacy.load('en', disable=['parser', 'ner', "vector", "tagger", "entity", "textcat"])

    def prepare_article(self, article):
        tokens = self.nlp(article)
        tokens_punctuation = [token.orth_ for token in tokens if token.is_punct]
        tokens_needed = "".join(tokens_punctuation)

        return tokens_needed

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        return lambda article: preprocess(self.decode(self.prepare_article(article)))


class PunctuationExtractor:
    def __init__(self):
        self.vectorizer = PunctuationStatisticsVectorizer()

    def extract_train(self, data):
        features = self.vectorizer.fit_transform(
            data.text)  # data.text c'est le contenu de la colone text donc c'est tous les database.db

        return features

    def extract_test(self, data):
        features = self.vectorizer.transform(data.text)

        return features

    def get_vectorizer(self):
        return self.vectorizer


def csrtomatrix(data, vocabulary):
    matrix = [[0 for _ in range(len(vocabulary))] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for j in range(len(vocabulary)):
            if data[i, j]:
                matrix[i][j] = data[i, j]
    return matrix


def create_csv_file(data, vocabulary, output_file):
    data = csrtomatrix(data, sorted(tuple(vocabulary.items())))
    table = pd.DataFrame(data)
    table.columns = sorted(list(vocabulary))
    table.index = [str(i) for i in range(len(data))]
    table.to_csv(output_file)


def main():
    parser = argparse.ArgumentParser(description='Punctation Statistics from csv file')
    parser.add_argument('-t', "--trainset", action='store',
                        default="../dataset/test_OK.csv",
                        help='Path to csv file '"[default: %(default)s]")
    parser.add_argument('-o', "--output", action='store',
                        default='../dataset/features/result_extraction_punctuation.csv',
                        help='Path to csv file '"[default: %(default)s]")
    args = parser.parse_args()
    working_file = args.trainset
    output_file = args.output
    data = pd.read_csv(working_file)
    extractor = PunctuationExtractor()
    features = extractor.extract_train(data)
    create_csv_file(features, extractor.vectorizer.vocabulary_, output_file)


if __name__ == '__main__':
    main()
