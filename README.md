# project\_disaster\_responses

### Environments
- Python 3.9
- Pandas
- Flask, Flask-WTF
- Plotly
- Scikit-Learn
- SQLAlchemy
- NLTK

### Application Architect
Flask + Bootstrap + SQLite + Machine Learning Model

The app use Python, Flask Framework as backend. 
A customized Bootstrap 5 theme as frontend UI. 
Data layer uses SQLite and CSV. 
A machine learning model (classification, multiple output class) is trained by Scikit-learn. (With NLTK processing inside)

### What is this app about?
The app allow user to use a machine learning model by inputting a disaster responses type of message, and get output of possible categories estimated by the model. 
The app also provide some interactive visualization about the dataset and model. 




### How to run the app
```bash
cd PATH_OF_PROJECT_ROOT
source vene/bin/active
python app.py
```

### File Structure
```bash
- (Project Root)
---- static
-------- static/assets (frontend UI files)
-------- static/csv (raw dataset)
-------- static/db (sqlite file)
-------- static/machine_learning_models (encoded models)
---- templates (jinja templates)
- app.py (app main file)
- process_data.py (ETL pipelines file)
- train_classifer.py (model training file)
- ...
```


### ETL Pipelines
to run ETL:
```bash
python process_data.py static/csv/messages.csv static/csv/categories.csv static/db/DisasterRes.db
```


### Train Machine Learning Model
Under project root directory
```bash
python train_classifier.py static/db/DisasterRes.db
```




### About the trained model
The model dumped is used with hyper-parameters tuning of, 
```python
'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
'features__text_pipeline__tfidf__use_idf': (True, False),
'clf__estimator__n_neighbors': [4, 5, 6]
```

results of an overall accuracy score 0.939.
The encoded (serialized) file size \> 100mb, cannot git to github. 
To reproduce this model. Need to run the training locally. 
It depends on the machine’s speed. It took my laptop almost an hour to train this. 
My version can be downloaded here: https://www.dropbox.com/s/o2iap7gud8yjywd/released_model.pkl?dl=0



### Visualization
The app provides three visualization plot. 
1. Genre Distribution (on /plotly page)
2. Category Distribution (on /plotly page)
3. Term Frequency Analysis (on /tf page)

For TF Analysis page, interactive form is added. User can select any category for result. 


### Known bugs:
If a category has no instances. e.g category: child\_alone. 
It would throw ValueError because empty text vocabulary, which means cannot process further analysis, then the app would break. 
To fix, add an error handling for such case, assemble a dummy word count dictionary. However, for illustration purpose, I would skip fixing this bug. To reproduce this error, select a category that contains nothing (or to be nothing after NLTK tokenizer (normalize / stop words remove / etc…) processing. 

Error Logs Below for future reference: 
```bash
[category]: child_alone
length of category: 11
Analyzing Most Common Terms Found in "child_alone" ...
[2021-10-31 00:43:34,002] ERROR in app: Exception on /tf [POST]
Traceback (most recent call last):
  File "PATH_OF_PROJECT_ROOT/venv/lib/python3.9/site-packages/flask/app.py", line 2073, in wsgi_app
    response = self.full_dispatch_request()
  File "PATH_OF_PROJECT_ROOT/venv/lib/python3.9/site-packages/flask/app.py", line 1518, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "PATH_OF_PROJECT_ROOT/venv/lib/python3.9/site-packages/flask/app.py", line 1516, in full_dispatch_request
    rv = self.dispatch_request()
  File "PATH_OF_PROJECT_ROOT/venv/lib/python3.9/site-packages/flask/app.py", line 1502, in dispatch_request
    return self.ensure_sync(self.view_functions[rule.endpoint])(**req.view_args)
  File "PATH_OF_PROJECT_ROOT/app.py", line 175, in tf
    wc_dict = _get_category_top_words(df, category)
  File "PATH_OF_PROJECT_ROOT/app.py", line 233, in _get_category_top_words
    cv_fit = cv.fit_transform(corpus)
  File "PATH_OF_PROJECT_ROOT/venv/lib/python3.9/site-packages/sklearn/feature_extraction/text.py", line 1330, in fit_transform
    vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary_)
  File "PATH_OF_PROJECT_ROOT/venv/lib/python3.9/site-packages/sklearn/feature_extraction/text.py", line 1220, in _count_vocab
    raise ValueError(
ValueError: empty vocabulary; perhaps the documents only contain stop words
```
Updated: this bug fixed. Update with a error handling. 




### About /tf page (Term Frequency Analysis)
The page use CounterVectorizer to compute word count. 
To be more specifically of term frequency context, might consider use TfidfTransformer or TfidfVectorizer to get TFIDF values and put together with word counts. 
However, for visualization illustration purpose, I would use CounterVectorizer. 
For a more ‘pro’ version, consider plotting {word, count of the word, TFIDF of the word} as dataset. Make TFIDF value to show when user hovering on the word. 