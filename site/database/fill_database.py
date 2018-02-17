#!/usr/bin/python
# -*- coding: utf-8 -*-

import sqlite3 as lite
con = lite.connect('articles')
with con:
    cur = con.cursor()
    for index in range(1,41):
        articles = "article{}".format(index)
        f = open(articles, encoding="utf8")
        article= f.readlines()
        article = " ".join(article)
        cur.execute("INSERT INTO Articles VALUES({},{},{})".format(int(index), int(2), article))
