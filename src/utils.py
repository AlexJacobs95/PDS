import pandas as pd


def saveMatrixAsCSV(matrix, columnNames, filename):
    dataframe = pd.DataFrame(matrix, columns=columnNames)
    saveCSV(dataframe, filename)


def saveCSV(dataframe, filename):
    dataframe.to_csv("../dataset/features/%s" % filename, index_label='index')
