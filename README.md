# Disaster Response Pipeline Project

This project builds an ETL pipeline to process disaster response messages and their categories. Then uses a ML pipeline to build a model to predict the category of messages, which can be entered by the user in a web UI.

## Dependencies

To run the project you need `python3` and `pip` installed. The full list of dependencies can be found in the `requirements.txt`.

## Installation

Clone the project

```
git clone https://github.com/luchev/disaster-response-pipeline
```

Enter project directory

```
cd disaster-response-pipeline
```

Set up a virtual environment

```
python3 -m venv venv
```

Activate the virtual invironment

```
source venv/bin/activate
```

Install requirements

```
pip install -r requirements.txt
```

## Usage

### Using pre-trained model

The project comes with wrangled data and pre-trained models. If you want to use the pre-trained models you can just run the webserver inside the app directory:

```
cd app
python3 run.py
```

Then go to http://0.0.0.0:3001/ to see the running project.

### Building custom model

If you want to train your own models on custom dataset you will have to update the model path in `app/run.py:34`

- To run ETL pipeline that cleans data and stores in database
    `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv <Database-Name>.db`

- To run ML pipeline that trains classifier and saves
    `python3 models/train_classifier.py <DatabasePath>.db <ModelPath>.pkl [ModelClassifier]`

There are 5 classifiers you can use:
- RandomForest (default)
- AdaBoost
- MLP
- KNeighbours
- DecisionTree

## Examples

Process the sample data using the ETL (executed in the root of the project)

```
python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

Train a new KNeighbours model (executed in the root of the project)

```
python3 models/train_classifier.py data/DisasterResponse.db models/new-kneighbours.pkl kneighbours
```

## Acknowledgements

This project is developed as part of [Udacity](https://www.udacity.com/)'s Data Science Nanodegree Program

The data is provided by [Figure Eight](https://appen.com/)

## License

This project is licensed under the terms of the MIT license.
