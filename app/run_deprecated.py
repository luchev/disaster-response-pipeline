import json
import plotly
import pandas as pd
import pickle

from operator import itemgetter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load models
try:
    modelAdaBoost = pickle.load(open('../models/adaboost.pkl', 'rb'))
except:
    pass
try:
    modelRandomForest = pickle.load(open('../models/randomforest.pkl', 'rb'))
except:
    pass
try:
    modelKNeighbours = pickle.load(open('../models/kneighbours.pkl', 'rb'))
except:
    pass
try:
    modelDecisionTree = pickle.load(open('../models/decisiontree.pkl', 'rb'))
except:
    pass
# modelMlp = pickle.load(open('../models/mlp.pkl', 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    categories = df.drop(
        ['id', 'message', 'original', 'genre'], axis=1).sum()
    category_names = [ x.replace('_', ' ').capitalize() for x in categories.index]
    category_counts = list(categories)
    
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Messages by category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'categoryorder': 'total descending',
                }
            },
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids,
            graphJSON=graphJSON, message_count=df.shape[0],
            category_count=len(category_names))


def predict_for_html(model, message):
    classification_labels = model.predict([message])[0]
    classification_results = list(zip(df.columns[4:], classification_labels))
    classification_results = sorted(classification_results, key=itemgetter(0))
    classification_results = sorted(classification_results, key=itemgetter(1), reverse=True)

    return classification_results

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    results = []

    try:
        results.append(
            ('K Neighbours', predict_for_html(modelKNeighbours, query)))
    except:
        pass
    try:
        results.append(
            ('Ada Boost', predict_for_html(modelAdaBoost, query)))
    except:
        pass
    try:
        results.append(
            ('Random Forest', predict_for_html(modelRandomForest, query)))
    except:
        pass
    try:
        results.append(
            ('Decision Tree', predict_for_html(modelDecisionTree, query)))
    except:
        pass

    # This will render the go.html 
    return render_template(
        'go.html',
        query=query,
        results=results,
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
