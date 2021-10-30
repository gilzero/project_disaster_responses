"""
Example of minial script to reuse the released model and do classification of a message instance.
"""


# import neccessary packages
import pickle
import numpy as np
from train_classifier import tokenize
from train_classifier import StartingVerbExtractor

# load from model pickle file
model = pickle.load(open('static/machine_learning_models/released_model.pkl', 'rb'))

# make test independent variable for classification/prediction
message = 'I need to eat something. Do you have burger? So Hungry.'

# match the np array shape
test_1 = np.array([message])


# classify / predict
result = model.predict(test_1)

print(result)


# To test multiple message, just make numpy array of mutiple messages
# msg1 = 'hello world'
# msg2 = 'hungry and cold'
# to_test = np.array([msg1, msg2])
# model.predict(to_test)