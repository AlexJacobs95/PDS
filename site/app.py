from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/game')
def game():
    return render_template('game.html')


@app.route('/checkAnswer')
def checkAnswer():
    raise NotImplementedError


def getAnswer(article_id):
    raise NotImplementedError
