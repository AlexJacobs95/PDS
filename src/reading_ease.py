from textstat.textstat import textstat
import pandas as pd
import numpy as np


def readability_score(df):
    lst_sc = []
    for i in df['text']:
        lst_sc.append(textstat.flesch_reading_ease(i))
    return np.transpose(np.matrix(lst_sc))


if __name__ == '__main__':
    df = pd.read_csv("../dataset/test_OK.csv")
    print(readability_score(df).shape)
