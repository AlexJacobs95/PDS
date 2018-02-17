import pandas as pd
import numpy as np
import itertools 
from all_features import * 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


class NaiveBayes : 

	def __init__(self, train_features, test_features) : 
		self.multinomial_NB = MultinomialNB() 
		self.bernoulli_NB = BernoulliNB()
		#self.gaussian_NB = GaussianNB()

	def fit(self, train_features, df_train) : 
		self.multinomial_NB.fit(train_features, df_train.code)
		self.bernoulli_NB.fit(train_features, df_train.code)
		#self.gaussian_NB.fit(train_features, df_train.code)
	
	def predict(self, test_features) : 
		multinomial_pred = self.multinomial_NB.predict(test_features) 
		bernoulli_pred = self.bernoulli_NB.predict(test_features)
		#gaussian_pred = self.gaussian_NB.predict(test_features)
		
		return multinomial_pred, bernoulli_pred

	def evaluate(self, df_test, pred_results) :
		score_list = [] 
		for i in range(len(pred_results)) :
			score_list.append(metrics.accuracy_score(df_test.code, pred_results[i])) 
			
			#print("accuracy:   %0.3f" % score)
		print(score_list)
		return score_list 

	"""def comparasionNB(self, scorelist): """


	def plot_confusion_matrix(self,cm,normalize=False, title='Naive Bayes'):
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

		classes = ["FAKE", "REAL"]
		plt.imshow(cm, interpolation='nearest')
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		print(thresh)
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

	def confusion_matrix(self, y_test, y_pred, t="") : 
		cnf_matrix = confusion_matrix(y_test.code, y_pred)
		np.set_printoptions(precision=2)
		plt.figure()
		self.plot_confusion_matrix(cnf_matrix, normalize=True,title=t)
		plt.show()


	"""def confusion (self, y_test, y_pred) : 
		labels = ['FAKE', 'REAL']
		cm = confusion_matrix(y_test.code, y_pred)
		print(cm)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm)
		plt.title('Naive Bayes')
		fig.colorbar(cax)
		ax.set_xticklabels([''] + labels)
		ax.set_yticklabels([''] + labels)
		plt.xlabel('Predicted')
		plt.ylabel('True')
		plt.show()"""


if __name__ == '__main__':

	dataframe_train = pd.read_csv("../dataset/train_80.csv")
    dataframe_test = pd.read_csv("../dataset/test_20.csv")
	
	train_features, test_features = return_all_features()
	clf = NaiveBayes(train_features, test_features)
	clf.fit(train_features, dataframe_train)
	m_prediction, b_prediction = clf.predict(test_features)
	
	clf.evaluate(dataframe_test, [m_prediction, b_prediction]) 
	
	clf.confusion_matrix(dataframe_test, m_prediction, "MultinomialNB")
	clf.confusion_matrix(dataframe_test, b_prediction, "BernoulliNB")
	#clf.confusion_matrix(dataframe_test, g_prediction, "GaussianNB")
	



	


