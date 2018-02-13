from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import time
import spacy
from tqdm import tqdm

def csrtomatrix(data, vocabulary):
    matrix = [[0 for _ in range(len(vocabulary))] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for j in range(len(vocabulary)):
            if data[i, j]:
                matrix[i][j] = data[i, j]
    return matrix


def create_csv_file(data, vocabulary, output_file):
    data = csrtomatrix(data, sorted(tuple(vocabulary.items())))
    table = pd.DataFrame(data)
    table.columns = sorted(list(vocabulary))
    table.index = [str(i) for i in range(len(data))]
    table.to_csv(output_file)

if __name__ == '__main__':
    all_data = pd.read_csv("../dataset/balancedtest.csv")
    res_tokens = []
    parser = spacy.load("en", disable=['parser', 'ner'])
    for data in tqdm(all_data.text):
        tokens = parser(data)
        token_pronouns = [token.text for token in tokens if token.tag_ in ["PRP", "PRP$", "WP", "WP$"]]
        res_tokens.append(" ".join(token_pronouns))
    count_pronoum_vect = CountVectorizer()
    pronoun_csr_matrix = count_pronoum_vect.fit_transform(res_tokens)

    # create_csv_file(pronoun_csr_matrix, count_pronoum_vect.vocabulary_, '../dataset/features/result_extraction_pronoum.csv')
