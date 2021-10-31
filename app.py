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
# Load DB
engine = create_engine('sqlite:///static/db/DisasterRes.db')
df = pd.read_sql_table('DisasterResponse', engine)
print(f"df.shape: {df.shape}")

# Get Category Names
cat_names = list(df.iloc[:, -36:].columns)
cat_names = [x.replace("_", ' ').title() for x in cat_names]
print(f"📈 cat_names:\n {cat_names}")

# options object for drop down selection in UI
options = [('all', 'All (Default)')] + list(
    zip(list(df.iloc[:, -36:].columns), [x.replace("_", ' ').title() for x in cat_names]))


# Define Forms
class MessageForm(FlaskForm):
    message = StringField('Message')
    submit = SubmitField('Submit')


class CategoryForm(FlaskForm):
    category = SelectField(
        'Select a Category to See Results:',
        choices=options
    )
    submit = SubmitField('Submit')


# Application Router

@app.route('/')
@app.route('/index')
def index():
    # General stats (top 10 categories info)
    names = list(df.iloc[:, -36:].sum().sort_values(ascending=False).head(10).index)
    top_cats_stats = []
    for n in names:
        title = n.replace('_', ' ').title()
        percent = f"{round(df[df[n] == 1].shape[0] / df.shape[0] * 100, 2)}%"
        mean_length = round(df[df[n] == 1]['message'].apply(lambda x: len(x)).describe()['mean'], 2)
        top_cats_stats.append((n, title, percent, mean_length))

    # Get 100 random sample messages from Dataset
    sample_msgs = list(df['message'].sample(n=100).values)

    # Total record value
    total = df.shape[0]

    # Average text length
    mean_length = round(df['message'].apply(lambda x: len(x)).describe()['mean'], 2)

    return render_template('index.html',
                           sample_len=len(sample_msgs), sample_msgs=sample_msgs,
                           total=total, mean_length=mean_length,
                           top_cats_stats=top_cats_stats, top_cats_stats_len=len(top_cats_stats))


@app.route('/plotly')
def plotly():
    # Two Plots Genre bar chart, category bar chart

    # ===== Genre Chart Handling ===== #
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

    # Assemble the figure object
    genre_fig = {'data': genre_data, 'layout': genre_layout}

    # jsonify the figure object for html/js parsing
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
        'title': 'Distribution of Message Categories',
        'xaxis': {'title': ''},
        'yaxis': {'title': 'Numbers'},
        'autosize': True
    }

    # Assemble the figure object
    cat_fig = {'data': cat_data, 'layout': cat_layout}

    # jsonify the figure object for html/js parsing
    cat_fig_json = json.dumps(cat_fig, cls=PlotlyJSONEncoder)

    return render_template('plotly.html', genre_fig_json=genre_fig_json, cat_fig_json=cat_fig_json)


@app.route('/model', methods=["GET", "POST"])
def model():
    print(f"[df.shape]: {df.shape}")

    # Get the message parameter value from POST. If message parameter is not POSTed, make it an empty string.
    if request.form.get('message'):
        message = request.form.get('message')
    else:
        message = ''

    print(f"[message]: {message}")

    # Allow user to input a message for classification labeling
    form = MessageForm()

    # Run the model to predict/classify:
    # load from model pickle file
    model = pickle.load(open('static/machine_learning_models/released_model.pkl', 'rb'))

    # convert message string to numpy array shape
    classification_input = np.array([message])

    # classify / predict
    classification_result = model.predict(classification_input)
    print(f"[classification_result]: {classification_result}")

    # Parse categories
    # if user does not enter any message text. make a dummy list
    # otherwise, assign to result variable

    if len(message) < 1:
        result = [0 for i in range(36)]
    else:
        result = list(classification_result[0])

    cats = dict(zip(cat_names, result))

    return render_template('model.html', form=form, template='form-template', message=message, cats=cats)


@app.route('/tf', methods=["GET", "POST"])
def tf():
    print('Viewing /tf page')

    # Get the category parameter value from POST. If category parameter is not POSTed, make it to default all.
    if request.form.get('category'):
        category = request.form.get('category')
    else:
        category = 'all'

    print(f"[category]: {category}")

    # Allow user to input a category
    form = CategoryForm()

    # Compute word count
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

    # Assemble the figure object
    tf_fig = {'data': tf_data, 'layout': tf_layout}

    # jsonify the figure object for html/js parsing
    tf_fig_json = json.dumps(tf_data, cls=PlotlyJSONEncoder)

    return render_template('tf.html', tf_fig_json=tf_fig_json, form=form, template='form-template', category=category)


@app.route('/about')
def about():
    return render_template('about.html')


# Private functions
def _get_category_top_words(df, category='all', size=30):
    # df: original df
    # category: column name: e.g 'weather_related', 'food'
    # size: top how many? e.g 20 = top 20 (most common 20 words)
    #
    # return a top word count dictionary

    print(f'Analyzing Most Common Terms Found in "{category}" ...')

    # Double check category name valid. if not in column name,
    # make it to all
    names = df.iloc[:, -36:].columns.to_list()

    if category not in names:
        print(f'User does not specify a category. Using All Messages by default.')
        category = 'all'
        messages = df['message']
    else:
        # subset the category message
        messages = df[df[category] == 1]['message']

    # Assemble a corpus
    corpus = list(messages.values)

    # initialize a CountVectorizer object
    cv = CountVectorizer(tokenizer=tokenize)

    # exception handle
    try:
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

        # convert to dictionary
        res = wc_df.sort_values(by=['number'], ascending=[False]).head(size)['number'].to_dict()
    except ValueError as e:
        # catching fit_transform. If the vocabulary is empty, then assemble a dummy 'NoRecord' word
        print(f"Exception: [ValueError]: \n{e}")
        res = {'NoRecord': 0}

    return res


if __name__ == '__main__':
    # use terminal 'python app.py' to run. (do not use 'flask run' in this case, otherwise pickle wont load properly.)
    # pickle dump has to remember the context where it dumped, it was under __main__. So to reload back, it has to
    # be under __main__ also. However 'flask run' will ignore code in '__main__'. Therefore, in order to properly load
    # pickle file (the model file), should use 'python app.py' method.
    # details about 'app.run() vs flask run'.  https://www.twilio.com/blog/how-run-flask-application
    # In case that want to use 'flask run' (or deploy on PaaS that user doesnt has options to run python, e.g heroku),
    # then need to rewrite the code, can make tokenize as a standlone module, with train_classifier import the
    # custom tokenize module, so when dumping out, it 'remembers' the context for external. When load back, also import
    # that custom tokenize, so that dump out and load back from same context.
    app.run()
