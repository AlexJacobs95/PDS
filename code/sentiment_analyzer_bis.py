import pandas as pd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

class SentimentExtractor : 

	def extract(self, data) : 
		output = []
		for news in data.text : 
			news_count = 0 
			blob = TextBlob(news,analyzer=NaiveBayesAnalyzer())
			output.append([blob.sentiment.p_pos,blob.sentiment.p_neg])
	
		return output

	def store_tocsv(self, results) : 
		df = pd.DataFrame(results)
		df.fillna('', inplace=True)
		df.to_csv('sentiment.csv')
		

if __name__ == '__main__': 
	data = pd.read_csv("../dataset/balancedtest_bis.csv")
	s_extractor = SentimentExtractor() 
	results = s_extractor.extract(data)
	s_extractor.store_tocsv(results)
	