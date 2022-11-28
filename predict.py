import joblib
from flask import Flask, redirect, url_for, render_template, request


def predict(x):
    loaded_model = joblib.load('model')
    result = loaded_model.predict(x)
    print(result)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('predictWebsite.html')


app.run()
