import numpy as np
import pandas as pd
from scipy import sparse

def return_all_features() : 
	
	tfidf_train = sparse.load_npz('../features/tfidf_train_features.npz')
    tfidf_test = sparse.load_npz('../features/tfidf_test_features.npz')

    punctuation_train = sparse.load_npz('../features/punctuations_train_features.npz')
    punctuation_test = sparse.load_npz('../features/punctuations_test_features.npz')

    pronouns_train = sparse.load_npz('../features/pronouns_train_features.npz')
    pronouns_test = sparse.load_npz('../features/pronouns_test_features.npz')

    text_counts_train = sparse.load_npz('../features/text_count_train_features.npz')
    text_counts_test = sparse.load_npz('../features/text_count_test_features.npz')

    readability_train = sparse.load_npz('../features/readablity_train_features.npz')
    readability_test = sparse.load_npz('../features/readablity_test_features.npz')

    sentiment_train = sparse.load_npz('../features/sentiment_train_features.npz')
    sentiment_test = sparse.load_npz('../features/sentiment_test_features.npz')

    train_features = sparse.hstack([tfidf_train, sentiment_train, pronouns_train])
    test_features = sparse.hstack([tfidf_test, sentiment_test, pronouns_test])

	return train_features, test_features 

if __name__ == '__main__': 
	return_all_features()