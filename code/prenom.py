import pandas as pd
from textblob import TextBlob


class PronounExtractor:
    """ Does a part-of-speech tagging to extract all pronouns
         Returns a list of number of pronouns per article """

    def extract(self, article):
        output = []
        for news in data.text:
            count = 0
            blob = TextBlob(news)
            for sentence in blob.sentences:
                for tag in sentence.tags:
                    if (tag[1] == 'PRP'):
                        count += 1
            output.append(count)
        return output


if __name__ == '__main__':
    data = pd.read_csv("../dataset/balancedtest_bis.csv")
    p_extractor = PronounExtractor()
    results = p_extractor.extract(data.text)
    print(results)
