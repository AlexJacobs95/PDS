import pandas as pd
from extractFeatures import *
import numpy as np
from time import time
from tqdm import *
import matplotlib.pyplot as plt

dataframe_train = pd.read_csv("../dataset/train_80.csv")

test = pd.read_csv("../dataset/test_20.csv")
dataframe_train = dataframe_train.reindex(np.random.permutation(dataframe_train.index))

size = dataframe_train.shape[0]

results = [] # contient les différents temps d'extraction
sizes = []

for percent in tqdm([0.05, 1, 5, 15, 25, 50, 75, 100]):

    to_keep = int((percent/100) * size)
    data = dataframe_train.iloc[0:to_keep]

    print(data.shape[0])

    sizes.append(data.shape[0])

    start = time()

    extractor = PunctuationExtractor()
    punctuation_train = extractFeatureWithVectorizer(extractor, 'punctuation_tfidf', data, train=True)

    extractor = PronounsExtractor()
    pronouns_train = extractFeatureWithVectorizer(extractor, 'pronouns_tfidf', data, train=True)

    extractor = TfidfExtractor(ngram=1)
    tfidf_train = extractFeatureWithVectorizer(extractor, 'tfidf_' , data, train=True)

    results.append(time()-start)


plt.figure()
plt.title("Temps d'extraction des features en fonction de la taille du dataset")
plt.xlabel("Taille du dataset")
plt.ylabel("Temps en secondes")
plt.plot(sizes, results)
plt.savefig("../figures/extraction_time.png")
plt.show()


