import argparse
import string # pour avoir liste de ponctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import time


class PunctuationStatisticsVectorizer(CountVectorizer):

    def __init__(self):
        super(PunctuationStatisticsVectorizer, self).__init__()

    def prepare_article(self, article):

        punctuation_list = list(string.punctuation)
        additional_punctuation = ['``', '--', '\'\'']
        punctuation_list.extend(additional_punctuation)
        article = article.replace("\\r\\n"," ")
        for char in  article:
            if char not in punctuation_list:
                article = article.replace(char,"")
        return article

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        return lambda article : preprocess(self.decode(self.prepare_article(article)))

# def extract_articles(data_file):
#
#     list_articles = []
#     data_file_df = pd.read_csv(data_file, names=["code","text"])
#     for i in range(1,len(data_file_df.index)):
#         list_articles.append(data_file_df.iloc[i]["text"])
#     return list_articles


class PunctuationExtractor:
    def __init__(self):
        self.punctuation_statistics_vect = PunctuationStatisticsVectorizer()

    def extract_train(self, data):
        print("Extracting Punctuation...")
        t0 = time.time()
        features = self.punctuation_statistics_vect.fit_transform(data.text) # data.text c'est le contenu de la colone text donc c'est tous les articles
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




def main():
    # parser = argparse.ArgumentParser(description='Punctation Statistics from csv file')
    # parser.add_argument('-t', "--trainset", action='store', default=None, help=('Path to csv file '"[default: %(default)s]"))
    # args = parser.parse_args()
    # working_file = args.trainset
    # list_articles = extract_articles(working_file)
    data = pd.read_csv("../dataset/train_bis.csv")
    extractor = PunctuationExtractor()
    features = extractor.extract_train(data)
    # print(punctuation_statistics_matrix)
    # print(punctuation_statistics_vect.vocabulary_)

if __name__ == '__main__':
    main()
