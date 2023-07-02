import random

from flask import Flask, send_from_directory, session, request, redirect, make_response, render_template

from Timer import timer

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

admin_username = "admin"
admin_password = "password@123"


@app.route('/<path:path>')
@timer
def send_static(path):
    path = path if path else 'index.html'
    return send_from_directory('static', path), request.path, request.remote_addr, '304'


@app.route('/admin')
@timer
def admin():
    key = request.cookies.get('key')
    if len(session) == 0:
        resp = make_response(render_template('message.html', message="Un-Authenticated Request"), 403)
        return resp, '/admin', request.remote_addr, '403'

    if int(key) == session['signed_key']:
        return render_template('message.html',
                               message="This Page is for Admins Only"), '/admin', request.remote_addr, '200'
    else:
        session.pop("signed_key", None)
        resp = make_response(render_template('message.html', message="Un-Authenticated Request"), 403)
        return resp, '/admin', request.remote_addr, '403'


@app.route('/login', methods=["GET", "POST"])
@timer
def login():
    if request.method == "GET":
        return send_from_directory('static', '/login.html'), '/login', request.remote_addr, '304'
    else:
        username = request.form['username']
        password = request.form['password']
        if username == admin_username and password == admin_password:
            key = random.randrange(1, 10000)
            resp = make_response(redirect('/admin'))
            resp.set_cookie('key', str(key))
            session['signed_key'] = key
            return resp, '/login', request.remote_addr, '200'
        else:
            session.pop('signed_key', None)
            resp = make_response(render_template('message.html', message='Authentication failed'), 401)
            return resp, '/login', request.remote_addr, '401'


if __name__ == "__main__":
    app.run("127.0.0.1", 5000)
