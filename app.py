import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, request
from flask import render_template
import json
from plotly.utils import PlotlyJSONEncoder
from pprint import pp
import plotly.express as px
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import os
import pickle
import numpy as np

from train_classifier import tokenize
from train_classifier import StartingVerbExtractor

app = Flask(__name__)

# Application Settings
# A secret key is required to use CSRF. form submission.
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

# Application Reusable Objects
## Load DataFrame
engine = create_engine('sqlite:///DisasterRes.db')
df = pd.read_sql_table('DisasterResponse', engine)
print(f"df.shape: {df.shape}")

# Category Names
cat_names = list(df.iloc[:, -36:].columns)
cat_names = [x.replace("_", ' ').title() for x in cat_names]
print(f"📈 cat_names:\n {cat_names}")


# Define Forms
class MessageForm(FlaskForm):
    message = StringField('Message')
    submit = SubmitField('Submit')


class PlotlyForm(FlaskForm):
    message = StringField('Message')
    submit = SubmitField('Submit')


# Application Router handling


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/plotly', methods=["GET", "POST"])
def plotly():
    return render_template('plotly.html')


@app.route('/model', methods=["GET", "POST"])
def model():
    print(f"df.shape: {df.shape}")

    if request.form.get('message'):
        message = request.form.get('message')
    else:
        message = ''

    print(f"[message]: {message}")
    print(f"length of message: {len(message)}")

    # allow user to input a message for classification labeling
    form = MessageForm()

    # Run the model to predict/classify:
    ## load from model pickle file
    model = pickle.load(open('static/machine_training_models/released_model.pkl', 'rb'))

    ## convert message string to numpy array shape
    classification_input = np.array([message])

    ## classify / predict
    classification_result = model.predict(classification_input)
    print(f"[classification_result]: {classification_result}")

    # parse categories
    # if user does not enter any message text. make a dummy list
    # otherwise, assign to result variable

    if len(message) < 1:
        result = [0 for i in range(36)]
        # print(type(result))
    else:
        result = list(classification_result[0])
        # print(type(result))

    cats = dict(zip(cat_names, result))

    return render_template('model.html', form=form, template='form-template', message=message, cats=cats)


if __name__ == '__main__':
    # use terminal python app.py to run. (do not use 'flask run' in this case, otherwise pickle wont load.)
    # app.run() vs flask run.  https://www.twilio.com/blog/how-run-flask-application
    app.run()
