import pandas as pd
import numpy as np
import spacy
import time
import string


class PronounExtractor:
    """ Does a part-of-speech tagging to extract all categories of pronouns treated by spacy
    Returns a n x len(pronoun listq) matrix of values for each category 
    each line contains the relative frequency of each pronouns
    Convert the results in csv file
    """

    def __init__(self):
        self.nlp = spacy.load('en')
        self.personal_pronoun_list = ["i", "you", "she", "he", "it", "we", "they"]
        self.possessive_pronoun_list = ["mine", "yours", "his", "hers", "ours", "theirs"]
        self.wh_personal_pronoun_list = ["what", "who", "whom"]
        self.wh_possessive_pronoun_list = ["whose", "whosever"]

    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def extract(self, article, p_list, p_tag):
        res = [0 for _ in range(len(p_list))]
        print(len(res))
        article = self.remove_punctuation(article)
        size = len(article)
        doc = self.nlp(article)

        for token in doc:
            if token.tag_ == p_tag:
                if token.text.lower() in p_list:
                    res[p_list.index(token.text.lower())] += 1

        res = [val / size for val in res]

        return res


    def transform(self, data):
        personal_results = []
        possessive_results = []
        wh_personal_results = []
        wh_possessive_results = []

        i = 1
        for article in data:
            #print(i)
            personal_results.append(self.extract(article,self.personal_pronoun_list, "PRP"))
            possessive_results.append(self.extract(article,self.possessive_pronoun_list, "PRP$"))
            wh_personal_results.append(self.extract(article,self.wh_personal_pronoun_list, "WP"))
            wh_possessive_results.append(self.extract(article,self.wh_possessive_pronoun_list, "WP$"))
            i += 1
        return np.vstack(personal_results) , np.vstack(possessive_results), np.vstack(wh_personal_results), np.vstack(wh_possessive_results)    
  
if __name__ == '__main__':
    
    dataframe_test = pd.read_csv("/home/prateeba/Desktop/BA3/Pds/Datasets/Datasheets/second.csv")
    extractor = PronounExtractor()
    start = time.time()
    
    r1, r2, r3, r4 = extractor.transform(dataframe_test.text)
    extract_time = time.time() - start
    print("extract time: %0.3fs" % extract_time)
    
    df = pd.DataFrame(r1, columns=["i", "you", "she", "he", "it", "we", "they"])
    df.to_csv("personal_pronouns_feature.csv")

    df = pd.DataFrame(r2, columns=["mine", "yours", "his", "hers", "ours", "theirs"])
    df.to_csv("possessive_pronouns_feature.csv")

    df = pd.DataFrame(r3, columns=["what", "who", "whom"])
    df.to_csv("wh_personal_pronouns_feature.csv")

    df = pd.DataFrame(r4, columns=["whose", "whosever"])
    df.to_csv("wh_possessive_pronouns_feature.csv") 

    
