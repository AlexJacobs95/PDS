import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import scipy.sparse as sparse


class NeuralNetwork:
    def __init__(self, inputSize, numLabels):
        self.model = Sequential()
        self.model.add(Dense(units=100, activation='relu', input_dim=inputSize))
        self.model.add(Dense(units=numLabels, activation='softmax'))
        self.model.compile(loss='categorial_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, articlesTrain, labelsTrain):
        self.model.fit(articlesTrain, labelsTrain, epochs=5, batch_size=32)

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
    features = None
    for file_path in getFeatureFilePaths():
        if 'test' in file_path and useTrainDataset:
            continue
        elif 'train' in file_path and not useTrainDataset:
            continue

        feature = sparse.load_npz(file_path)

        if features is None:
            features = feature
        else:
            features = sparse.vstack(features, feature)

    labels = loadLabels(useTrainDataset)
    return features, labels


if __name__ == '__main__':
    trainNews, trainLabels = loadDataset(useTrainDataset=True)
    testNews, testLabels = loadDataset(useTrainDataset=False)
    neuralNetwork = NeuralNetwork(inputSize=trainNews.shape[1], numLabels=trainLabels.shape[1])
    neuralNetwork.fit(trainNews, trainLabels)

    loss_and_metrics = neuralNetwork.evaluate(testNews, testLabels)
    print(loss_and_metrics)
