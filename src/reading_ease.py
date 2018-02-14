from textstat.textstat import textstat
import pandas as pd

def Readability_score(csv):
	df = pd.read_csv(csv)
	lst_sc=[]
	for i in df['text']:
		lst_sc.append(textstat.flesch_reading_ease(i))
	return lst_sc

if __name__ == '__main__':
	print(Readability_score("test_OK.csv"))
