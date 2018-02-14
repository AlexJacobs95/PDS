from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time
import spacy
import string


punctuations = string.punctuation
nlp = spacy.load('en', disable=['parser', 'ner'])

i = 0


def spacy_tokenizer(sentence):
    global i
    i += 1
    print(i)
    tokens = nlp(sentence)
    tokens = [tok.text for tok in tokens if (tok.text not in punctuations)]
    return tokens


class TfidfExtractor:
    def __init__(self, ngram):
        #self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1, ngram),
        #                                  tokenizer=spacy_tokenizer)

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1, ngram))

    def extract_train(self, data):
        features = self.vectorizer.fit_transform(data.text)
        return features

    def extract_test(self, data):
        features = self.vectorizer.transform(data.text)
        return features

    def get_vectorizer(self):
        return self.vectorizer
