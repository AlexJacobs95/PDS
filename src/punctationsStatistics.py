import argparse
import string  # pour avoir liste de ponctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import time
from spacy.lang.en import English as EN
import spacy


class PunctuationStatisticsVectorizer(CountVectorizer):

    def __init__(self):
        super(PunctuationStatisticsVectorizer, self).__init__()
        self.nlp = spacy.load('en')

    # def prepare_article(self, article):
    #
    #     punctuation_list = list(string.punctuation)
    #     additional_punctuation = ['``', '--', '\'\'']
    #     punctuation_list.extend(additional_punctuation)
    #     article = article.replace("\\r\\n"," ")
    #     for char in  article:
    #         if char not in punctuation_list:
    #             article = article.replace(char,"")
    #     return article

    def prepare_article(self, article):
        tokens = self.nlp(article)
        tokens_punctuation = [token.orth_ for token in tokens if token.is_punct]
        token_pronouns = [token.text for token in tokens if token.tag_ in ["PRON", "PRP$", "WP", "WP$"]]
        tokens_needed = " ".join(tokens_punctuation + token_pronouns)

        return tokens_needed

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        return lambda article: preprocess(self.decode(self.prepare_article(article)))


class PunctuationExtractor:
    def __init__(self):
        self.punctuation_statistics_vect = PunctuationStatisticsVectorizer()

    def extract_train(self, data):
        print("Extracting Punctuation...")
        t0 = time.time()
        features = self.punctuation_statistics_vect.fit_transform(
            data.text)  # data.text c'est le contenu de la colone text donc c'est tous les articles
        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

        return features

    def extract_test(self, data):
        print("Extracting Punctuation...")
        t0 = time.time()
        features = self.punctuation_statistics_vect.transform(data.text)
        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

        return features


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
                        default="../dataset/balancedtest.csv",
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
    create_csv_file(features, extractor.punctuation_statistics_vect.vocabulary_, output_file)


if __name__ == '__main__':
    main()
