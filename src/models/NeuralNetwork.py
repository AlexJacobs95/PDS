from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


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


def loadDataset(filepath):
    """
    Reads the dataset (articles and labels) from the file and returns two sets of data:
        - A training set (to train the model): trainNews, trainLabels
        - A test set (to test the model): testNews, testLabels
    """
    # X_train, X_test, y_train, y_test = train_test_split(
    #     articles, labels, test_size=0.33, random_state=42)
    raise NotImplementedError


if __name__ == '__main__':
    trainNews, trainLabels, testNews, testLabels = loadDataset('../dataset/features_small.csv')
    neuralNetwork = NeuralNetwork(inputSize=trainNews.shape[1], numLabels=trainLabels.shape[1])
    neuralNetwork.fit(trainNews, trainLabels)


    loss_and_metrics = neuralNetwork.evaluate(testNews, testLabels)
    print(loss_and_metrics)
