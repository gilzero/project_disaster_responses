import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, request
from flask import render_template
import json
from plotly.utils import PlotlyJSONEncoder
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from train_classifier import tokenize
from train_classifier import StartingVerbExtractor

app = Flask(__name__)

# Application Settings
# A secret key is required to use CSRF. form submission.
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

# Application Reusable Objects
## Load DataFrame
engine = create_engine('sqlite:///static/db/DisasterRes.db')
df = pd.read_sql_table('DisasterResponse', engine)
print(f"df.shape: {df.shape}")

# Category Names
cat_names = list(df.iloc[:, -36:].columns)
cat_names = [x.replace("_", ' ').title() for x in cat_names]
print(f"📈 cat_names:\n {cat_names}")
options = [('all', 'All (Default)')] + list(
    zip(list(df.iloc[:, -36:].columns), [x.replace("_", ' ').title() for x in cat_names]))


# Define Forms
class MessageForm(FlaskForm):
    message = StringField('Message')
    submit = SubmitField('Submit')


class TfForm(FlaskForm):
    message = StringField('Message')
    category = SelectField(
        'Select a Category to See Results:',
        choices=options
    )
    submit = SubmitField('Submit')


# Application Router handling


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/plotly')
def plotly():
    # Three Plots Genre bar chart, category bar chart, word count per category chart

    # ===== Gener Chart Handling ===== #
    genre_x = [x.title() for x in df['genre'].value_counts().index.tolist()]
    genre_y = df['genre'].value_counts().values.tolist()

    genre_data = [{
        'type': 'bar',
        'x': genre_x,
        'y': genre_y
    }]

    genre_layout = {
        'title': '',
        'xaxis': {'title': 'Genre'},
        'yaxis': {'title': 'Numbers'},
        'autosize': True

    }

    # assemble the figure object
    genre_fig = {'data': genre_data, 'layout': genre_layout}

    # jsonfy the figure object for html/js parsing
    genre_fig_json = json.dumps(genre_fig, cls=PlotlyJSONEncoder)

    # ===== Category Chart Handling ===== #
    count_cats = df.iloc[:, -36:].sum().sort_values(ascending=False)
    count_cats_k = [x.replace('_', ' ').title() for x in count_cats.index.to_list()]
    count_cats_v = list(count_cats.values)

    cat_data = [{
        'type': 'bar',
        'x': count_cats_k,
        'y': count_cats_v
    }]

    cat_layout = {
        'title': '',
        'xaxis': {'title': 'Categories'},
        'yaxis': {'title': 'Numbers'},
        'autosize': True
    }

    # assemble the figure object
    cat_fig = {'data': cat_data, 'layout': cat_layout}

    # jsonfy the figure object for html/js parsing
    cat_fig_json = json.dumps(cat_fig, cls=PlotlyJSONEncoder)

    return render_template('plotly.html', genre_fig_json=genre_fig_json, cat_fig_json=cat_fig_json)


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


@app.route('/tf', methods=["GET", "POST"])
def tf():
    print('Viewing /tf page')

    if request.form.get('category'):
        category = request.form.get('category')
    else:
        category = 'all'

    print(f"[category]: {category}")
    print(f"length of category: {len(category)}")

    # allow user to input a category
    form = TfForm()

    # compute word count
    wc_dict = _get_category_top_words(df, category)
    print(wc_dict)

    # ===== TF Chart Handling ===== #
    tf_words = list(wc_dict.keys())
    tf_counts = list(wc_dict.values())

    tf_data = [{
        'type': 'bar',
        'x': tf_words,
        'y': tf_counts
    }]

    tf_layout = {
        'title': '',
        'xaxis': {'title': 'Terms'},
        'yaxis': {'title': 'Counts'},
        'autosize': True
    }

    # assemble the figure object
    tf_fig = {'data': tf_data, 'layout': tf_layout}

    # jsonfy the figure object for html/js parsing
    tf_fig_json = json.dumps(tf_data, cls=PlotlyJSONEncoder)

    return render_template('tf.html', tf_fig_json=tf_fig_json, form=form, template='form-template', category=category)


# Private functions
def _get_category_top_words(df, category='all', size=30):
    # df: original df
    # category: column name: e.g 'weather_related', 'food'
    # size: top how many? e.g 20 = top 20 (most common 20 words)
    #
    # return a top word count dictionary

    print(f'Analyzing Most Common Terms Found in "{category}" ...')

    # double check category name valid. if not in column name,
    # make it to all
    names = df.iloc[:, -36:].columns.to_list()

    if category not in names:
        print(f'Category {category} Not Found. Using All Messages.')
        category = 'all'
        messages = df['message']
    else:
        # subset the category message
        messages = df[df[category] == 1]['message']

    # assemble a corpus
    corpus = list(messages.values)

    # initialize count vectorized object
    cv = CountVectorizer(tokenizer=tokenize)

    # fit the transformer to get each token (word)
    cv_fit = cv.fit_transform(corpus)

    # all words (feature names)
    word_list = cv.get_feature_names_out()

    # count of each word (feature names) in array shape
    count_list = np.asarray(cv_fit.sum(axis=0))[0]

    # (word count dictionary) Concat feature names and count value together.
    wc_dict = dict(zip(word_list, count_list))

    # assemble to a DataFrame for computation.
    wc_df = pd.DataFrame.from_dict(wc_dict, orient='index', columns=['number'])

    return wc_df.sort_values(by=['number'], ascending=[False]).head(size)['number'].to_dict()


if __name__ == '__main__':
    # use terminal python app.py to run. (do not use 'flask run' in this case, otherwise pickle wont load properly.)
    # details about 'app.run() vs flask run'.  https://www.twilio.com/blog/how-run-flask-application
    app.run()
