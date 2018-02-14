from src.TextCountExtractor import *
from src.TfidfExtractor import *
from src.punctuationsExtractor import *
from src.pronoumExtractor import *
from src.reading_ease import *
from src.sentiment_analyzer import *
from scipy import sparse
import time

if __name__ == '__main__':
    train = pd.read_csv("../dataset/train_bis.csv")
    test = pd.read_csv("../dataset/test_OK.csv")

    start_total = time.time()

    # train features
    print("extracting tfidf from train...")
    start = time.time()
    features = TfidfExtractor(1).extract_train(train)
    sparse.save_npz('../dataset/features/tfifd_train_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting punctuation from train...")
    start = time.time()
    features = PunctuationExtractor().extract_train(train)
    sparse.save_npz('../dataset/features/punctuations_train_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting pronouns from train...")
    start = time.time()
    features = PronounsExtractor().extract_train(train)
    sparse.save_npz('../dataset/features/pronouns_train_features', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting text_count from train...")
    start = time.time()
    features = sparse.csr_matrix(TextCountExtractor().transform(train))
    sparse.save_npz('../dataset/features/text_count_train_features', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting readability from train...")
    start = time.time()
    features = sparse.csr_matrix(readability_score(train))
    sparse.save_npz('../dataset/features/readablity_train_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting sentiment from train...")
    start = time.time()
    features = sparse.csr_matrix(SentimentExtractor('../resources/emotion.csv').words_classifier(train))
    sparse.save_npz('../dataset/features/sentiment_train_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    # test features
    print("extracting tfidf from test...")
    start = time.time()
    features = TfidfExtractor(1).extract_test(test)
    sparse.save_npz('../dataset/features/tfifd_test_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting punctuation from test...")
    start = time.time()
    features = PunctuationExtractor().extract_test(test)
    sparse.save_npz('../dataset/features/punctuations_test_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting pronouns from test...")
    start = time.time()
    features = PronounsExtractor().extract_test(test)
    sparse.save_npz('../dataset/features/pronouns_test_features', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting text_count from test...")
    start = time.time()
    features = sparse.csr_matrix(TextCountExtractor().transform(test))
    sparse.save_npz('../dataset/features/text_count_test_features', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting readability from test...")
    features = sparse.csr_matrix(readability_score(test))
    sparse.save_npz('../dataset/features/readablity_test_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    print("extracting sentiment from test...")
    start = time.time()
    features = sparse.csr_matrix(SentimentExtractor('../resources/emotion.csv').words_classifier(test))
    sparse.save_npz('../dataset/features/sentiment_test_features.npz', features)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)

    extract_total = time.time() - start_total
    print("extract time: %0.3fs" % extract_total)
