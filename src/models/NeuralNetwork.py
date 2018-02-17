import os

from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
import pandas as pd
import scipy.sparse as sparse
import keras

from sklearn import preprocessing
import numpy as np


class NeuralNetwork:
    def __init__(self, inputSize, numLabels, model=None):
        """
        self.model = Sequential()
        self.model.add(Dense(units=500, activation='relu', input_dim=inputSize))
        self.model.add(Dense(units=numLabels, activation='softmax'))
        sgd = SGD(lr=1e-6)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        """
        # Cette architecture est inspirée de ce site
        # https://www.kaggle.com/jacklinggu/tfidf-to-keras-dense-neural-network
        if model is None:
            self.model = Sequential()
            self.model.add(Dense(500, input_dim=inputSize))
            self.model.add(Activation('relu'))
            self.model.add(Dense(100))
            self.model.add(Activation('relu'))
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))
            self.model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['acc'])
        else:
            self.model = model
        print(self.model.summary())

    def fit(self, articlesTrain, labelsTrain):
        #tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        #self.model.fit(articlesTrain, labelsTrain, callbacks=[tbCallBack])
        self.model.fit(articlesTrain, labelsTrain)
        self.model.save('keras_model.h5')

    def predict(self, articles):
        return self.model.predict(articles)

    def evaluate(self, articlesTest, labelsTest):
        loss_and_metrics = self.model.evaluate(articlesTest, labelsTest)
        return loss_and_metrics


def getFeatureFilePaths():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    featuresDir = os.path.join(currentDir, '..', '..', 'features')
    return [os.path.join(featuresDir, filename) for filename in os.listdir(featuresDir)]


def getDatasetDir():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    datasetDir = os.path.join(currentDir, '..', '..', 'dataset')
    return datasetDir


def loadLabels(useTrainDataset=True):
    file_name = 'train_80.csv' if useTrainDataset else 'test_20.csv'
    df = pd.read_csv(os.path.join(getDatasetDir(), file_name))
    return df.code


def loadDataset(useTrainDataset=True):
    """
    Reads the dataset (articles and labels) from the file and returns two sets of data:
        - A training set (to train the model): trainNews, trainLabels
        - A test set (to test the model): testNews, testLabels
    :param useTrainDataset: if True, use the training dataset, otherwise use the test dataset.
    """
    features = []
    for file_path in getFeatureFilePaths():
        # if 'readablity' in file_path:
        #     continue
        if 'tfidf' in file_path and not ('tfidf_500' in file_path):
            continue

        if 'test' in file_path and useTrainDataset:
            continue
        elif 'train' in file_path and not useTrainDataset:
            continue

        feature = sparse.load_npz(file_path)
        features.append(feature)

    # We stack all the features, then convert to CSR to allow slicing row-wise,
    # which is required to batches in the neural network
    featureMatrix = sparse.hstack(features).tocsr()
    labels = loadLabels(useTrainDataset)
    return featureMatrix, labels

def shuffleData(inputData, labels):
    idx = np.random.permutation(len(labels))
    return inputData[idx], labels[idx]

if __name__ == '__main__':
    
    print("Loading training dataset")
    trainNews, trainLabels = loadDataset(useTrainDataset=True)
    print("Standardizing training data")
    scaler = preprocessing.MaxAbsScaler().fit(trainNews)
    trainNews = scaler.transform(trainNews)
    print("Shuffling data")
    trainNews, trainLabels = shuffleData(trainNews, trainLabels)
    print("Loading testing dataset")
    testNews, testLabels = loadDataset(useTrainDataset=False)
    print("Standardizing testing data")
    testNews = scaler.transform(testNews)
    print("Real: ", len([x for x in testLabels if x]))
    print("Fake: ", len([x for x in testLabels if not x]))
    
    print("Building neural network")
    fromfile = False
    if fromfile is False:
        print(trainNews.shape)
        neuralNetwork = NeuralNetwork(inputSize=trainNews.shape[1], numLabels=1) 
        print("Training neural network")
        neuralNetwork.fit(trainNews, trainLabels)
    else:
        neuralNetwork = NeuralNetwork(inputSize=None, numLabels=None, model=load_model('keras_model.h5'))

    print("Evaluating neural network performance on training set")
    loss_and_metrics = neuralNetwork.evaluate(testNews, testLabels)
    print(loss_and_metrics)
