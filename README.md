# project\_disaster\_responses

### Environments and Libraries
- Python 3.9
- Pandas
- Flask, Flask-WTF
- Plotly
- Scikit-Learn
- SQLAlchemy
- NLTK
- Bootstrap 5

### What is this app about?
The app allow user to use a machine learning model by inputting a disaster responses type of message, and get output of possible categories estimated by the model. 
The app also provide some interactive visualization about the dataset and model. 

### Application Architect
Flask + Bootstrap + SQLite + Plotly + Machine Learning Model

The app use Python, Flask Framework as backend. 
A customized Bootstrap 5 theme as frontend UI. 
Data layer uses SQLite and CSV. 
A machine learning model (classification, multiple output class) is trained by Scikit-learn. (With NLTK processing inside)


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
to run ETL, under project root directory.
```bash
python process_data.py static/csv/messages.csv static/csv/categories.csv static/db/DisasterRes.db
```


### Train Machine Learning Model
Under project root directory.
```bash
python train_classifier.py static/db/DisasterRes.db
```


### About the trained model
The model dumped is used with hyper-parameters tuning of:
```python
'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
'features__text_pipeline__tfidf__use_idf': (True, False),
'clf__estimator__n_neighbors': [4, 5, 6]
```

Results of an overall accuracy score 0.939.
The encoded (serialized) file size \> 100mb, cannot git to github. 
To reproduce this model. Need to run the training locally. 
It depends on the machine’s speed. It took my laptop almost an hour to train this. 
My version can be downloaded here: https://www.dropbox.com/s/o2iap7gud8yjywd/released\_model.pkl?dl=0

### Visualization
The app provides three visualization plot. 
1. Genre Distribution (on /plotly page)
2. Category Distribution (on /plotly page)
3. Term Frequency Analysis (on /tf page)

For TF Analysis page, interactive form is added. User can select any category for result. 

### About /tf page (Term Frequency Analysis)
The page use CounterVectorizer to compute word count. 
To be more specifically of term frequency context, might consider use TfidfTransformer or TfidfVectorizer to get TFIDF values and put together with word counts. 
However, for visualization illustration purpose, I would use CounterVectorizer. 
For a more ‘pro’ version, consider plotting {word, count of the word, TFIDF of the word} as dataset. Make TFIDF value to show when user hovering on the word. 

### Application Screenshot
![alt screenshot](static/screenshots/screenshot_index.png)

![alt screenshot](static/screenshots/screenshot_model.png)

![alt screenshot](static/screenshots/screenshot_tf.png)

![alt screenshot](static/screenshots/screenshot_tf_dropdown.png)

![alt screenshot](static/screenshots/screenshot_plotly.png)

![alt screenshot](static/screenshots/screenshot_about.png)