"""
Rename to process_data.py if necessary.


To read back from db

from sqlalchemy import create_engine
engine = create_engine('sqlite:///DisasterRes.db')
pd.read_sql("select * from DisasterResponse", con=engine)
"""

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # messages.head()

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # categories.head()

    # merge datasets
    df = pd.merge(messages, categories)
    # df.head()

    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # categories.head()

    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [x.split('-')[0] for x in row.values[0].tolist()]
    # print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames
    # categories.head()

    # convert category values to 0 or 1
    # [fixed] fix this. there are some records are 2.
    # everything > value 0 make them to 1
    # otherwise later model classification will find 0,1,2
    categories = categories.apply(lambda x: (x.str.split('-').str.get(1).astype('int64') > 0).astype('int64'))
    # categories = categories.apply(lambda x: x.str.split('-').str.get(1))

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    df.duplicated().sum()

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # check number of duplicates
    df.duplicated().sum()

    return df


def save_data(df, database_filename):
    # engine = create_engine('sqlite:///InsertDatabaseName.db')
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
