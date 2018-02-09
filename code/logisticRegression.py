from sklearn import linear_model
from TfidfExtractor import *
import pandas as pd
import time

dataframe = pd.read_csv("../../dataset/train_bis.csv")

classifier = linear_model.LogisticRegression(verbose=True)
features = TfidfExtractor.extract(dataframe, 2)

start = time.time()
print("Training SVM Classifier...")
classifier.fit(features, dataframe.code)
print("Training time : " + time.time() - start + "s")