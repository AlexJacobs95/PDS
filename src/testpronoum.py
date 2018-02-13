from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import time
import spacy
import string
import argparse

punctuations = string.punctuation
nlp = spacy.load('en', disable=['parser', 'ner'])


punctuations_list = [elem for elem in string.punctuation]
prononus = ["i", "you", "she", "he", "it", "we", "they"] + ["mine", "yours", "his", "hers", "ours", "theirs"] + ["what", "who", "whom"] + ["whose", "whosever"]




def spacy_tokenizer(sentence):
    tokens = nlp(sentence)
    tokens = [tok.text for tok in tokens if (tok.text in punctuations or tok.tag_ in ["PRP", "PRP$", "WP", "WP$"])]
    return tokens


class Extractor:
    def __init__(self):
        self.vectorizer = CountVectorizer(vocabulary = prononus + punctuations_list, tokenizer=spacy_tokenizer)


    def extract_train(self, data):
        print("Extracting...")
        t0 = time.time()
        features = self.vectorizer.fit_transform(data.text)
        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

        return features

    def extract_test(self, data):
        print("Extracting..")
        t0 = time.time()
        features = self.vectorizer.transform(data.text)
        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

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
    extractor = Extractor()
    features = extractor.extract_train(data)
    print(extractor.vectorizer.vocabulary_)
    create_csv_file(features, extractor.vectorizer.vocabulary_, output_file)

if __name__ == '__main__':
    main()