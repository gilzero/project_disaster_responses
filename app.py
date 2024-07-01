import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from flask import Flask, request, render_template, flash
from plotly.utils import PlotlyJSONEncoder
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from train_classifier import tokenize
from train_classifier import StartingVerbExtractor

app = Flask(__name__)

# Application Settings
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(32)

# Database setup
DB_PATH = 'static/db/DisasterRes.db'
MODEL_PATH = 'static/machine_learning_models/released_model.pkl'

def load_data():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found at {DB_PATH}")

    engine = create_engine(f'sqlite:///{DB_PATH}')

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='DisasterResponse'"))
            if result.fetchone() is None:
                raise ValueError("'DisasterResponse' table not found in the database")

        df = pd.read_sql_table('DisasterResponse', engine)
        print(f"Successfully read table. Shape: {df.shape}")
        return df
    except SQLAlchemyError as e:
        raise SQLAlchemyError(f"Database error: {str(e)}")

def load_model():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
    except (FileNotFoundError, AttributeError, ImportError) as e:
        print(f"Error loading model: {str(e)}")
        print("Creating a simple fallback model.")
        # Create a simple fallback model with text preprocessing
        X = df.message.values
        Y = df.iloc[:, 4:].values
        fallback_model = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)))
        ])
        fallback_model.fit(X, Y)
        return fallback_model

# Load data
df = load_data()

# Load model with fallback
model = load_model()

# Get Category Names
cat_names = [x.replace("_", ' ').title() for x in df.iloc[:, 4:].columns]
print(f"📈 cat_names:\n {cat_names}")

# Options object for drop down selection in UI
options = [('all', 'All (Default)')] + list(zip(df.iloc[:, 4:].columns, cat_names))

# Define Forms
class MessageForm(FlaskForm):
    message = StringField('Message')
    submit = SubmitField('Submit')

class CategoryForm(FlaskForm):
    category = SelectField('Select a Category to See Results:', choices=options)
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    category_stats = df.iloc[:, 4:].sum().sort_values(ascending=False).head(10)
    top_cats_stats = [
        (name, name.replace('_', ' ').title(),
         f"{round(count / df.shape[0] * 100, 2)}%",
         round(df[df[name] == 1]['message'].str.len().mean(), 2))
        for name, count in category_stats.items()
    ]

    sample_msgs = df['message'].sample(n=100).tolist()
    total = df.shape[0]
    mean_length = round(df['message'].str.len().mean(), 2)

    return render_template('index.html',
                           sample_len=len(sample_msgs), sample_msgs=sample_msgs,
                           total=total, mean_length=mean_length,
                           top_cats_stats=top_cats_stats, top_cats_stats_len=len(top_cats_stats))

@app.route('/plotly')
def plotly():
    genre_counts = df['genre'].value_counts()
    genre_data = [{
        'type': 'bar',
        'x': genre_counts.index.tolist(),
        'y': genre_counts.values.tolist()
    }]

    genre_layout = {
        'title': 'Distribution of Message Genres',
        'xaxis': {'title': 'Genre'},
        'yaxis': {'title': 'Count'},
        'autosize': True
    }

    genre_fig = {'data': genre_data, 'layout': genre_layout}
    genre_fig_json = json.dumps(genre_fig, cls=PlotlyJSONEncoder)

    count_cats = df.iloc[:, 4:].sum().sort_values(ascending=False)
    cat_data = [{
        'type': 'bar',
        'x': [x.replace('_', ' ').title() for x in count_cats.index],
        'y': count_cats.values.tolist()
    }]

    cat_layout = {
        'title': 'Distribution of Message Categories',
        'xaxis': {'title': ''},
        'yaxis': {'title': 'Count'},
        'autosize': True
    }

    cat_fig = {'data': cat_data, 'layout': cat_layout}
    cat_fig_json = json.dumps(cat_fig, cls=PlotlyJSONEncoder)

    return render_template('plotly.html', genre_fig_json=genre_fig_json, cat_fig_json=cat_fig_json)

@app.route('/model', methods=["GET", "POST"])
def model_page():
    form = MessageForm()
    message = request.form.get('message', '')

    if message:
        try:
            classification_input = np.array([message])
            classification_result = model.predict(classification_input)
            result = list(classification_result[0])
        except Exception as e:
            print(f"Error in model prediction: {str(e)}")
            flash("An error occurred during prediction. Using fallback prediction.", "warning")
            result = [0] * len(cat_names)
    else:
        result = [0] * len(cat_names)

    cats = dict(zip(cat_names, result))
    return render_template('model.html', form=form, template='form-template', message=message, cats=cats)

@app.route('/tf', methods=["GET", "POST"])
def tf():
    form = CategoryForm()
    category = request.form.get('category', 'all')

    wc_dict = _get_category_top_words(df, category)

    tf_data = [{
        'type': 'bar',
        'x': list(wc_dict.keys()),
        'y': list(wc_dict.values())
    }]

    tf_layout = {
        'title': f'Top Words in "{category.replace("_", " ").title()}" Category' if category != 'all' else 'Top Words in All Categories',
        'xaxis': {'title': 'Terms'},
        'yaxis': {'title': 'Counts'},
        'autosize': True
    }

    tf_fig_json = json.dumps(tf_data, cls=PlotlyJSONEncoder)

    return render_template('tf.html', tf_fig_json=tf_fig_json, form=form, template='form-template', category=category)

@app.route('/about')
def about():
    return render_template('about.html')

def _get_category_top_words(df, category='all', size=30):
    print(f'Analyzing Most Common Terms Found in "{category}" ...')

    if category not in df.columns and category != 'all':
        print(f'Invalid category. Using All Messages by default.')
        category = 'all'

    messages = df['message'] if category == 'all' else df[df[category] == 1]['message']

    cv = CountVectorizer(tokenizer=tokenize, stop_words='english')

    try:
        cv_fit = cv.fit_transform(messages)
        word_list = cv.get_feature_names_out()
        count_list = np.asarray(cv_fit.sum(axis=0))[0]
        word_counts = dict(zip(word_list, count_list))
        return dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:size])
    except ValueError as e:
        print(f"Error in word counting: {str(e)}")
        return {'NoRecord': 0}

if __name__ == '__main__':
    app.run(debug=True)