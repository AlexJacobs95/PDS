import argparse
import time

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('en', disable=['parser', 'ner', 'entity', "vector", "tagger", "textcat"])

pronouns = ['whom', 'her', 'you', 'yours', 'whose', 'theirs', 'our', 'itself',
            'they', 'my', 'us', 'he', 'herself', 'himself', 'themselves',
            'she', 'whoever', 'hers', 'yourselves', 'your', 'its', 'me',
            'yourself', 'what', 'we', 'his', 'myself', 'ourselves', 'i',
            'their', 'who', 'him', 'it']


def spacy_tokenizer(sentence):
    tokens = nlp(sentence)
    tokens = [tok.orth_ for tok in tokens if tok.text.lower() in pronouns]
    return tokens


class PronounsExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)

    def extract_train(self, data):
        features = self.vectorizer.fit_transform(data.text)
        return features

    def extract_test(self, data):
        try:
            features = self.vectorizer.transform(data.text)
        except AttributeError:  # Not a dataframe, just an article
            features = self.vectorizer.transform(data)
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
                        default='../dataset/features/result_extraction.csv',
                        help='Path to csv file '"[default: %(default)s]")
    args = parser.parse_args()
    working_file = args.trainset
    output_file = args.output
    data = pd.read_csv(working_file)
    extractor = PronounsExtractor()
    features = extractor.extract_train(data)
    create_csv_file(features, extractor.vectorizer.vocabulary_, output_file)


if __name__ == '__main__':
    main()
