import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf
from flask import Flask, redirect, url_for, render_template, request
import joblib

import os

training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

# the four expected results in the same order
target_data = np.array([[0], [1], [1], [0]], "float32")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])


def getRandomInt():
    return random.randint(0, 1)


def getRandomInput():
    return np.array([[getRandomInt() for i in range(2)]], "float32")


def getOutput(input):
    return input[0] ^ input[1]


def train(epochs):
    model.fit(training_data, target_data, epochs=epochs, verbose=2)

    joblib.dump(model, 'model')


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('trainingWebsite.html')


app.run()
