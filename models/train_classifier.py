#!python3

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import sys
import os
import re
import pickle
import pandas as pd
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])


def load_data(database_filename: str) -> (np.array, pd.DataFrame, [str]):
    """ Load and split training data from database

    Args:
        database_filename (str): Path to the database

    Returns:
        (messages, category_values, category_names)

        messages (List[str]): list of text messages
        category_values (pd.DataFrame): dataframe with the categories for each message as bools
        category_names (List[str]): list with the names of the categories in category_values
    """
    database_filename = re.sub('\.db$', '', database_filename)
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    database_name = os.path.basename(database_filename)
    df = pd.read_sql_table(database_name, engine).head(10)
    
    messages = df['message'].values
    categories = df.drop(['id', 'genre', 'message', 'original'], axis=1)

    return (messages, categories, categories.columns)

def tokenize(text: str) -> [str]:
    """ Tokenize text, by removing punctuation, lemmatizing and tokenizing it

    Args:
        text (str): Text to tokenize
    
    Returns:
        List[str]: List of clean tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url, 'url')

    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens]
    return tokens


def build_model(cpus: int = 1):
    """ Build a TF IDF model with Multi output classifier

    Args:
        cpus (int): Number of CPUs to use when fitting the model
            Default is 1. Set cpus to -1 if you want to use all available

    Returns:
        GridSearchCV: Multi output ML model using TF IDF
    """
    pipeline = Pipeline([
        ('vector', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('multiout', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'vector__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'vector__stop_words': (None, 'english'),
        'tfidf__sublinear_tf': (True, False),
        'tfidf__use_idf': (True, False),
        'multiout__estimator__leaf_size': (10, 30, 50),
        'multiout__estimator__weights': ('uniform', 'distance'),
        'multiout__estimator__n_jobs': [cpus],
        'multiout__n_jobs': [cpus],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model: Pipeline, X_test: np.array, Y_test: pd.DataFrame, category_names: str) -> None:
    """ Output model precision and f1 score for all categories to the stdout

    Args:
        model (Pipeline): Model to use when fitting the data
        X_test (List[str]): Input messages to predict categories for
        Y_test (pd.DataFrame): Actual categories for all messages
        category_names (List[str]): List with the category names
    """
    Y_predicted = model.predict(X_test)
    for index, column in enumerate(category_names):
        print('Category', index + 1, column + ':')
        print(classification_report(Y_predicted[:, index], Y_test[column].values))


def save_model(model: Pipeline, model_filepath: str) -> None:
    """ Serialize a ML model to a file
    
    Args:
        model (ML model): Any ML model (Pipeline or GridSearchCV for example)
        model_filepath (str): File path in which to serialize the model
    """
    model_filepath = re.sub('.pkl$', '', model_filepath)
    pickle.dump(model, open(model_filepath + '.pkl', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
