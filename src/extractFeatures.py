from TextCountExtractor import *
from TfidfExtractor import *
from punctuationsExtractor import *
from pronoumExtractor import *
from reading_ease import *
from sentiment_analyzer import *
from scipy import sparse
import pandas as pd
import time

if __name__ == '__main__':
    train = pd.read_csv("../dataset/train_80.csv")
    test = pd.read_csv("../dataset/test_20.csv")
    sentiment_data = pd.read_csv("../resources/emotion.csv")

    start_total = time.time()


    print("extracting tfidf from train...")
    start = time.time()
    extractor = TfidfExtractor(1)
    features = extractor.extract_train(train)
    sparse.save_npz('../features/tfidf_train_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)
    print("extracting tfidf from test...")
    start = time.time()
    features = extractor.extract_test(test)
    sparse.save_npz('../features/tfidf_test_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting punctuation from train...")
    start = time.time()
    extractor = PunctuationExtractor()
    features = extractor.extract_train(train)
    sparse.save_npz('../features/punctuations_train_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)
    print("extracting punctuation from test...")
    start = time.time()
    features = extractor.extract_test(test)
    sparse.save_npz('../features/punctuations_test_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)


    print("extracting pronouns from train...")
    start = time.time()
    extractor = PronounsExtractor()
    features = extractor.extract_train(train)
    sparse.save_npz('../features/pronouns_train_features', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)
    print("extracting pronouns from test...")
    start = time.time()
    features = extractor.extract_test(test)
    sparse.save_npz('../features/pronouns_test_features', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)



    print("extracting text_count from train...")
    start = time.time()
    features = sparse.csr_matrix(TextCountExtractor().transform(train))
    sparse.save_npz('../features/text_count_train_features', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)
    print("extracting text_count from test...")
    start = time.time()
    features = sparse.csr_matrix(TextCountExtractor().transform(test))
    sparse.save_npz('../features/text_count_test_features', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting readability from train...")
    start = time.time()
    features = sparse.csr_matrix(readability_score(train))
    sparse.save_npz('../features/readablity_train_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)
    print("extracting readability from test...")
    features = sparse.csr_matrix(readability_score(test))
    sparse.save_npz('../features/readablity_test_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting sentiment from train...")
    start = time.time()
    features = sparse.csr_matrix(SentimentExtractor(sentiment_data).words_classifier(train))
    sparse.save_npz('../features/sentiment_train_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)
    print("extracting sentiment from test...")
    start = time.time()
    features = sparse.csr_matrix(SentimentExtractor(sentiment_data).words_classifier(test))
    sparse.save_npz('../features/sentiment_test_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)


    extract_total = time.time() - start_total
    print("total extract time: %0.3fs" % extract_total)
