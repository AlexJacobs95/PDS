import json
import pandas as pd
import time


class ProfanityExtractor:
    """
    Extracts the relative frequency of profanities in each article
    """

    def __init__(self, dictionary):
        self.dictionary = json.load(open(dictionary))

    def count_profanities(self, article):
        count = 0
        for word in article.split():
            if word in self.dictionary:
                count += 1

        return count

    def extract(self, data):
        result = []
        print("Extracting profanities...")
        done = 0
        t0 = time.time()
        for article in data.text:
            result.append(self.count_profanities(article) / len(article.split()))
            done += 1
            print("Done : " + str(done) + '/' + str(len(data)))

        extract_time = time.time() - t0
        print("extract time: %0.3fs" % extract_time)

        return result


if __name__ == '__main__':
    dataset = pd.read_csv('../dataset/test_OK.csv')
    extractor = ProfanityExtractor('../resources/profanities.json')
    res = extractor.extract(dataset)
    print(res)
