import argparse
import string # pour avoir liste de ponctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class PunctationStatistics():
    pass

def extract_articles(data_file):

    dict_articles = {0:[], 1:[]}
    data_file_df = pd.read_csv(data_file, names=["code","text"])
    print(len(data_file_df.index))
    for i in range(len(data_file_df.index)):
        if data_file_df.iloc[i]["code"]=="0":
            dict_articles[0].append(data_file_df.iloc[i]["text"])
        else:
            dict_articles[1].append(data_file_df.iloc[i]["text"])
    return dict_articles
    
def main():
    parser = argparse.ArgumentParser(description='Punctation Statistics from a list of articles')
    parser.add_argument('-t', "--trainset", action='store', default=None, help=('Path to csv file '"[default: %(default)s]"))
    args = parser.parse_args()
    working_file = args.trainset
    dict_articles = extract_articles(working_file)

if __name__ == '__main__':
    main()
