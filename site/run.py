from app import *
import os

if __name__ == '__main__':
    app.secret_key = os.urandom(24)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0')
