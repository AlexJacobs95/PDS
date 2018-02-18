#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import sqlite3 as lite

articles = []
labels = []
ids = []

id = 0
for index in range(1,41):
        label = 1
        if index <=20:
            label = 0
        file = "article{}".format(index)
        f = open('../articles/' + file, encoding="utf8")
        article = f.readlines()
        article = " ".join(article)
        articles.append(article)
        labels.append(label)
        ids.append(id)
        id += 1

data = pd.DataFrame(data={'id': ids, 'label': labels, 'content': articles})
print(data)

con = lite.connect('database.db')
data.to_sql('Articles', con=con, index=False)




