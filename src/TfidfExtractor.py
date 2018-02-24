from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import time
import spacy
import string
import numpy as np
import matplotlib.pyplot as plt

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


def plot_tfidf_classfeats_h(dfs):
    """ Plot the data frames returned by the function plot_tfidf_classfeats(). """
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


class TfidfExtractor:
    def __init__(self, ngram, max_features=None):
        # self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1, ngram),
        #                                  tokenizer=spacy_tokenizer)

        self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=max_features, max_df=0.5,
                                          stop_words='english', ngram_range=(1, ngram))

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

    def top_tfidf_feats(self, row, features, top_n=25):
        """ Get top n tfidf values in row and return them with their corresponding feature names."""
        topn_ids = np.argsort(row)[::-1][:top_n]
        top_feats = [(features[i], row[i]) for i in topn_ids]
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        return df

    def top_mean_feats(self, Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
        """ Return the top n features that on average are most important amongst documents in rows
            indentified by indices in grp_ids. """
        if grp_ids:
            D = Xtr[grp_ids].toarray()
        else:
            D = Xtr.toarray()

        D[D < min_tfidf] = 0
        tfidf_means = np.mean(D, axis=0)
        return self.top_tfidf_feats(tfidf_means, features, top_n)

    def top_feats_by_class(self, Xtr, y, features, min_tfidf=0.1, top_n=25):
        ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
            calculated across documents with the same class label. '''
        dfs = []
        labels = np.unique(y)
        for label in labels:
            ids = np.where(y == label)
            feats_df = self.top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
            feats_df.label = label
            dfs.append(feats_df)
        return dfs


if __name__ == '__main__':
    TRAIN = pd.read_csv("../dataset/train_80.csv")
    extractor = TfidfExtractor(ngram=1, max_features=10000)
    y = TRAIN.code
    Xtr = extractor.extract_train(TRAIN)
    features = extractor.get_vectorizer().get_feature_names()

    res = extractor.top_feats_by_class(Xtr, y, features)
    plot_tfidf_classfeats_h(res)