# import libraries
import sys
import re
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# constants and reusable objects
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(data_file):
    '''
    Load the cleaned dataset from SQLite db file.

    :param data_file: SQLite db file.
    :return: a tuple of messages text array, categories array and list of column names
    '''
    # load data from database
    engine = create_engine(f"sqlite:///{data_file}")
    df = pd.read_sql("select * from DisasterResponse", con=engine)
    print(f"📈 df.shape: {df.shape}")

    # define features and label arrays
    # Independent (X) and Dependent (y) variables.
    # In this case, each instance of y is a multi-value array represents all possible categories
    X = df.message.values
    y = df.iloc[:, -36:].values

    # list of category column names
    cat_columns = list(df.iloc[:, -36:].columns)

    print(f"📈 X.shape: {X.shape}")
    print(f"📈 y.shape: {y.shape}")
    print(f"📈 cat_columns:\n {cat_columns}")

    return X, y, cat_columns


def build_model():
    '''
    Build a machine learning model with feature union pipelines and GridSearch tuning.

    :return: a model pipeline object
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    # Check possible hyper-parameters if necessary
    # for i in list(pipeline_new.get_params().keys()):
    #     print(i)

    # Tuning hyper-parameters for GridSearchCV
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        # 'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_neighbors': [4, 5, 6]
    }

    # Create GridSearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline


def train(X, y, cat_columns, model):
    '''
    Do the training of the model.

    :param X: Independent variables. The messages array.
    :param y: Dependent variable. The categories array.
    :param cat_columns: A list of category names.
    :param model: The model object.
    :return: A trained model.
    '''

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Fit the model
    start = timer()
    model.fit(X_train, y_train)
    end = timer()
    print(f"⌛ Time Processed Training: {round((end - start), 2)} seconds")

    # Output model test results
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"💡 Model Accuracy: {accuracy}")

    # Output classification_report related values
    for i in range(len(cat_columns)):
        precision, recall, fscore, support = score(y_test[:, i], y_pred[:, i], average='micro')
        print(
            f"⭐ [{cat_columns[i]}] => fscore: {round(fscore, 2)}, precision: {round(precision, 2)}, recall: {round(recall, 2)}")

    return model


def export_model(model):
    '''
    Export model as a pickle file. Encode (Serialize) the model object from memory into persistent storage.
    Hardcoded to this file path: static/machine_learning_models/released_model.pkl

    :param model: The trained model.
    :return:
    '''


    print("📦 Exporting Model...")
    pickle.dump(model, open('static/machine_learning_models/released_model.pkl', 'wb'))
    print("📦 Model Exported/Released. 'released_model.pkl' ✅")


def run_pipeline(data_file):
    '''
    The script's main execution logic flow.

    :param data_file: The cleaned SQLite db file.
    :return:
    '''

    X, y, cat_columns = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, cat_columns, model)  # train model pipeline
    export_model(model)  # save model


# Private Functions

def tokenize(text):
    '''
    private tokenizer to transform each text.
    As a NLP helper function including following tasks:
    - Replace URLs
    - Normalize text
    - Remove punctuation
    - Tokenize words
    - Remove stop words
    - Legmmatize words

    :param text: A message text.
    :return: cleaned tokens extracted from original message text.
    '''

    # print(f"original text: \n {text}")

    # replace urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    # in case after normalize/lemmatize, if there is no words, make a dummy element. otherwise StartingVerb breaks
    if len(tokens) < 1:
        tokens = ['none']

    # print(f"tokens: \n {tokens} \n\n")
    return tokens


# Custom Transformer
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    A custom transform for feature union.
    Check whether a message's starting word is a verb.
    '''

    def starting_verb(self, text):
        '''
        Internal helper function. Check if starting word is verb

        :param text: A message text.
        :return: 1 or 0 represents true or false.
        '''
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        '''
        Apply the lambda to the message column.
        :param X: Each datapoint
        :return: A dataframe represents if starting is verb
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline

