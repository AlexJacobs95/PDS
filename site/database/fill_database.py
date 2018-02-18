#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import sqlite3 as lite

articles = []
labels = []

for index in range(1,41):
        label = 1
        if index <=20:
            label = 0
        file = "article{}".format(index)
        f = open(file, encoding="utf8")
        article = f.readlines()
        article = " ".join(article)
        articles.append(article)
        labels.append(label)

data = pd.DataFrame(data={'label': labels, 'content': articles})
print(data)

con = lite.connect('articles')
data.to_sql('articles', con=con, index=True)




