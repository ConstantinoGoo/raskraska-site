from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Hello! Flask is working!</h1>'

if __name__ == '__main__':
    print("Starting server at http://127.0.0.1:8080")
    app.run(host='127.0.0.1', port=8080) 