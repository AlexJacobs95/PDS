import csv
import sys
import string
import numpy as np
import pickle
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.externals import joblib


csv.field_size_limit(sys.maxsize)


categories = {
			  1: "satire",
			  2: "hoax",
			  3: "propaganda",
			  4: "trusted"
			 }

class Article:
	def __init__(self, target, content):
		self.target = target
		self.content = content


class DataSet:
	def __init__(self,articles):
		self.data = [article.content for article in articles ]
		self.target = np.asarray([article.target for article in articles])

def create_dataset(csv_file):
	data = parseData(csv_file)
	print("Creating dataset")
	dataset= DataSet(data)
	print("Dataset created.")

	dataset_size_mb = size_mb(dataset.data)
	print("%d documents - %0.3fMB" % (
	    len(dataset.data), dataset_size_mb))
	print()

	return dataset

def extract_features_train(dataset, vectorizer):
	print("Extracting features")
	t0 = time()	
	features = vectorizer.fit_transform(dataset.data)
	extract_time = time() - t0
	print("extract time: %0.3fs" % extract_time)
	return features

def extract_features_test(dataset, vectorizer):
	print("Extracting features")
	t0 = time()	
	features = vectorizer.transform(dataset.data)
	extract_time = time() - t0
	print("extract time: %0.3fs" % extract_time)
	return features

def train(clf, X_train, y_train):
	print("Training classifier")
	t0 = time()
	clf.fit(X_train, y_train)
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
	return clf

def test(clf, X_test, y_test):
	t0 = time()
	pred = clf.predict(X_test)
	test_time = time() - t0
	print("test time:  %0.3fs" % test_time)

	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def clean_text(text):
	translator = str.maketrans('', '', string.punctuation)
	content = text.lower()
	content = content.translate(translator)
	return content.strip()

def parseData(file):
	dataset = []
	translator = str.maketrans('', '', string.punctuation)
	print("Parsing...")
	with open('fakenewsfiles/fakenewsfiles/' + file, newline='') as csvfile:
		articles = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in articles:
			row =' '.join(row).strip()
			if file == "balancedtest.csv":
				cat = int(row[0])
			else:
				cat = int(row[1])
			content = row[5:len(row) - 1]
			content = content.lower()
			content = content.translate(translator)
			dataset.append(Article(cat, content))

	print("Pasing done.")

	return dataset




##################################################################################


def main_train(ngram=1):
	data_train = create_dataset("train.csv")
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english',ngram_range=(1,ngram))
	X_train = extract_features_train(data_train, vectorizer)
	y_train = data_train.target
	clf = linear_model.LogisticRegression()
	trained_clf = train(clf, X_train, y_train)
	print("Dumping classifier and vectorizer...")
	joblib.dump(trained_clf, 'classifier.pkl')
	joblib.dump(vectorizer, 'vectorizer.pkl')
	print("Dumping done.")

def load_model():
	print("Loading classifier and Vectorizer...")
	clf = joblib.load('classifier.pkl')
	vectorizer = joblib.load('vectorizer.pkl')
	print("Loading done.")
	return clf, vectorizer

def main_test(clf, vectorizer):
	data_test = create_dataset("balancedtest.csv")
	X_test = extract_features_test(data_test, vectorizer)
	y_test = data_test.target
	test(clf, X_test, y_test)

def make_prediction(text, clf, vectorizer):
	text = clean_text(text)
	my_pred = clf.predict(vectorizer.transform([text]))[0]
	print("PREDICTION : " + str(categories[int(my_pred)]))
