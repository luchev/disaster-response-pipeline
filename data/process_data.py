#!python3

import re
import os
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """ Load 2 csv files as dataframes and join them into 1 dataframe
    
    Args:
        messages_filepath (str): path to the messages CSV file
        categories_filepath (str): path to the categories CSV file

    Returns:
        pd.DataFrame: The two files merged as one dataframe
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    return pd.merge(messages_df, categories_df, how='inner', on=['id'])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean and wrangle a dataframe

    Args:
        df (pd.DataFrame): dataframe to clean
    
    Returns:
        pd.DataFrame: dataframe where the categories column has been replaced
            by multiple boolean columns, one column per category
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in list(row)]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """ Save a dataframe to a database

    Args:
        df (pd.DataFrame): dataframe to save
        database_filename (str): database name, also used for the sql file name
    """
    database_filename = re.sub('\.db$', '', database_filename)
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    database_name = os.path.basename(database_filename)
    df.to_sql(database_name, engine, index=False) 


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
