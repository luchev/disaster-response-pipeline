# Disaster Response Pipeline Project

This project builds an ETL pipeline to process disaster response messages and their categories. Then uses a ML pipeline to build a model to predict the category of messages, which can be entered by the user in a web UI. Check it out on https://disasterresponsepipeline.herokuapp.com/

The UI shows what categories the model predicted for the inputted message, by using 4 different ML models. This allows for better predictions and comparison between the models.

Here's a few examples

![](https://i.imgur.com/cLP6DiS.png)

![](https://i.imgur.com/ZCq0icm.png)

![](https://i.imgur.com/Qv9fFX9.png)

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

The project comes with wrangled data and pre-trained models. If you want to use the pre-trained models you can just run the webserver:

```
python3 run.py
```

or

```
./runFlask.sh
```

Then go to http://0.0.0.0:3001/ to see the running project.

### Building custom model

If you want to train your own models on custom dataset you will have to update the model path in `app/run.py:34`

- To run ETL pipeline that cleans data and stores in database
    `python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv <Database-Name>.db`

- To run ML pipeline that trains classifier and saves
    `python3 models/train_classifier.py <DatabasePath>.db <ModelPath>.pkl [ModelClassifier]`

There are 4 classifiers you can use:
- RandomForest (default)
- AdaBoost
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
