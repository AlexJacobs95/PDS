class NeuralNetwork:
    def predict(self, articles):
        raise NotImplementedError

    def fit(self, articles, labels):
        raise NotImplementedError


def loadDataset(filepath):
    """
    Reads the dataset (articles and labels) from the file and returns two sets of data:
        - A training set (to train the model): trainNews, trainLabels
        - A test set (to test the model): testNews, testLabels
    """
    raise NotImplementedError


def analysePredictions(predictedLabels, realLabels):
    """
    Analyses how good the predictions are.
    This still has to be defined (accuracy, precision, MCSS, or something else?) and implemented.
    """
    raise NotImplementedError


if __name__ == '__main__':
    trainNews, trainLabels, testNews, testLabels = loadDataset('../dataset/test_OK.csv')
    neuralNetwork = NeuralNetwork()
    neuralNetwork.fit(trainNews, trainLabels)

    predictions = neuralNetwork.predict(testNews)
    analysePredictions(predictions, testLabels)
