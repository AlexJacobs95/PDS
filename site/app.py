import sqlite3
from flask import request
from flask import g
from flask import Flask, redirect
from flask import jsonify
from flask import render_template
from random import randint

app = Flask(__name__)

DATABASE = 'database/database.db'
ARTICLE_NUMBER = 40


def make_dicts(cursor, row):
    return dict((cursor.description[idx][0], value)
                for idx, value in enumerate(row))


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = make_dicts
    return db


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/game', methods=['GET', 'POST'])
def game():
    if request.method == 'GET':
        content = getArticleContent()
        return render_template('game.html', article_content=content)

    elif request.method == 'POST':
        return jsonify({'value': 'True'})


def getArticleContent():
    articleID = randint(0, ARTICLE_NUMBER - 1)
    query = "SELECT content FROM Articles WHERE id=" + str(articleID)
    info = query_db(query)
    content = info[0]['content']
    return content


def checkInDB(article_id, answer):
    query = "SELECT label" \
            "FROM Articles" \
            "WHERE id = " + str(article_id)
    label = query_db(query)

    return label == answer


def getAnswer(article_id):
    raise NotImplementedError
