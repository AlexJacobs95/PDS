import sqlite3
from flask import request
from flask import g
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import session
from flask import redirect
import random
from googletrans import Translator

app = Flask(__name__)

DATABASE = 'database/database.db'
ARTICLE_NUMBER = 40
NUMBER_OF_ROUNDS_PER_GAME = 5
TRANSLATOR = Translator()


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

        # Return True of the answer was correct, else return False
        player_answer = request.form['value']
        return_val = jsonify({'correct': checkIfCorrect(session["current_article"]['label'], player_answer),
                              'newArticleContent': session["current_article"]["content"], 'displayPopupFinish': False})

        if session["round"] <= NUMBER_OF_ROUNDS_PER_GAME-2:
            updateGame()
        else:
            return_val = jsonify({'correct': checkIfCorrect(session["current_article"]['label'], player_answer),
                                  'newArticleContent': session["current_article"]["content"], 'displayPopupFinish': True})
        return return_val


def initGame():
    session["game_articles_ids"] = genateRandomIds(NUMBER_OF_ROUNDS_PER_GAME)
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
    #Uncomment to translate articles
    # content = TRANSLATOR.translate(content, dest='fr').text
    return {'content': content, 'label': label}


def genateRandomIds(x):
    """
    Return x articles (id, content, label)
    """
    return random.sample(range(0, ARTICLE_NUMBER), x)
