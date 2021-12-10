# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 12:41:14 2021

@author: hp
"""

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import joblib

loaded_model=joblib.load(r"D:/ML_Project\model.pkl")
loaded_stop=joblib.load(r"D:/ML_Project\stopwords.pkl")
loaded_vec=joblib.load(r"D:/ML_Project\vectorizer.pkl")

app = Flask(__name__)

def classify(document):
    label = {0: 'non-hate', 1: 'hate'}
    X = loaded_vec.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    if proba >= 0.85:
        return label[0], proba
    else:
        return label[1], proba

class TweetForm(Form):
    tweet = TextAreaField('',[validators.DataRequired(),validators.length(min=15)])

@app.route(r'/')
def index():
    form = TweetForm(request.form)
    return render_template('tweetform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = TweetForm(request.form)
    if request.method == 'POST' and form.validate():
        tweet = request.form['tweet']
        y, proba = classify(tweet)
        print(proba)
        return render_template('results.html', content=tweet, prediction=y, probability=round(proba*100, 2))
    return render_template('tweetform.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)