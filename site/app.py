import sqlite3

import sys

import os
from flask import request
from flask import g
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import session
import random
from googletrans import Translator

app = Flask(__name__)
app.secret_key = 'super secret key'

DATABASE = 'database/database.db'
NUMBER_OF_ARTICLES = 40
NUMBER_OF_ROUNDS_PER_GAME = 5
TRANSLATOR = Translator()

AIisOn = True
if AIisOn:
    sys.path.insert(1, os.path.join(sys.path[0], '..', 'src'))
    from predictor import Predictor

    predictor = Predictor()


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
        initGame()
        print(session["round"])
        return render_template('game.html', article_content=session["current_article"]["content"])

    elif request.method == 'POST':
        # Returns True if the answer was correct, else return False
        player_answer = request.form['value']
        article, label = session["current_article"]["content"], session["current_article"]['label']

        if AIisOn:
            aiIsCorrect = predictor.predict(article) == label
        else:
            aiIsCorrect = False

        finishedGame = (session["round"] + 1 >= NUMBER_OF_ROUNDS_PER_GAME)

        if not finishedGame:
            updateGame()

        return jsonify({
            'correct': checkIfCorrect(label, player_answer),
            'aiCorrect': aiIsCorrect,
            'newArticleContent': session["current_article"]["content"],
            'displayPopupFinish': finishedGame
        })


def initGame():
    session["game_articles_ids"] = generateRandomIds(NUMBER_OF_ROUNDS_PER_GAME)
    session["round"] = 0
    session["current_article"] = getArticle(session["game_articles_ids"][session["round"]])


def updateGame():
    session["round"] += 1
    session["current_article"] = getArticle(session["game_articles_ids"][session["round"]])


def checkIfCorrect(real_label, answer):
    """
    Check if the answer and the label associated to the article are the same
    """
    return (real_label == 1 and answer == 'true') or (real_label == 0 and answer == 'false')


def getArticle(articleID):
    """
    Return the (content, label) of the article having id = articleID
    """
    query = "SELECT content, id, label FROM Articles WHERE id=" + str(articleID)
    info = query_db(query)
    content, label = info[0]['content'], info[0]['label']
    # Uncomment to translate articles
    # content = TRANSLATOR.translate(content, dest='fr').text
    return {'content': content, 'label': label}


def generateRandomIds(x):
    """
    Return x articles (id, content, label)
    """
    return random.sample(range(0, NUMBER_OF_ARTICLES), x)
