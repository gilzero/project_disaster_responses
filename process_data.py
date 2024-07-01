"""
Data ETL pipelines script.

This script loads data from CSV files, cleans and transforms the data,
and saves the result to a SQLite database.

python process_data.py static/csv/messages.csv static/csv/categories.csv static/db/DisasterRes.db

"""

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(messages_filepath, categories_filepath):
    """
    Load data from two CSV files and return a merged dataframe.

    Args:
    messages_filepath (str): Path to the messages CSV file.
    categories_filepath (str): Path to the categories CSV file.

    Returns:
    pandas.DataFrame: Merged dataset
    """
    try:
        # Check if files exist
        if not os.path.exists(messages_filepath):
            raise FileNotFoundError(f"Messages file not found: {messages_filepath}")
        if not os.path.exists(categories_filepath):
            raise FileNotFoundError(f"Categories file not found: {categories_filepath}")

        # Load messages dataset
        messages = pd.read_csv(messages_filepath)
        logger.info(f"Loaded messages data: {messages.shape}")

        # Load categories dataset
        categories = pd.read_csv(categories_filepath)
        logger.info(f"Loaded categories data: {categories.shape}")

        # Merge datasets
        df = pd.merge(messages, categories, on='id')
        logger.info(f"Merged data shape: {df.shape}")

        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def clean_data(df):
    """
    Clean and transform the dataset.

    Args:
    df (pandas.DataFrame): Merged dataset.

    Returns:
    pandas.DataFrame: Cleaned dataset.
    """
    try:
        # Create a dataframe of the 36 individual category columns
        categories = df['categories'].str.split(';', expand=True)

        # Extract new column names for categories
        row = categories.iloc[0]
        category_colnames = row.apply(lambda x: x.split('-')[0])
        categories.columns = category_colnames

        # Convert category values to 0 or 1
        for column in categories:
            categories[column] = categories[column].apply(lambda x: int(x.split('-')[1]) > 0).astype(int)

        # Replace categories column in df with new category columns
        df = df.drop('categories', axis=1)
        df = pd.concat([df, categories], axis=1)

        # Remove duplicates
        initial_duplicates = df.duplicated().sum()
        df = df.drop_duplicates()
        final_duplicates = df.duplicated().sum()
        
        logger.info(f"Removed {initial_duplicates - final_duplicates} duplicate rows")
        logger.info(f"Cleaned data shape: {df.shape}")

        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def save_data(df, database_filename):
    """
    Save the cleaned dataset to a SQLite database.

    Args:
    df (pandas.DataFrame): The cleaned dataset.
    database_filename (str): File path of SQLite database.
    """
    try:
        engine = create_engine(f"sqlite:///{database_filename}")
        df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
        logger.info(f"Data saved to {database_filename}")
    except Exception as e:
        logger.error(f"Error saving data to database: {str(e)}")
        raise

def main():
    """
    Main execution logic.
    Takes terminal arguments of messages CSV file, categories CSV file, and output SQLite db filepath.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        logger.info('Loading data...\n    MESSAGES: %s\n    CATEGORIES: %s',
                    messages_filepath, categories_filepath)
        df = load_data(messages_filepath, categories_filepath)

        logger.info('Cleaning data...')
        df = clean_data(df)
        
        logger.info('Saving data...\n    DATABASE: %s', database_filepath)
        save_data(df, database_filepath)
        
        logger.info('Cleaned data saved to database!')
    
    else:
        logger.error('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument.\n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()