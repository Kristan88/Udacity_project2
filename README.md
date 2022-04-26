# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
1.  Required libraries:

    - pandas
    - numpy
    - sqlalchemy
    - sklearn
    - re
    - pickle
    - nltk
    

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's direpyctory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

This work was done within the 2nd project of Udacity Nanodegree course. 
The goal was to build an entire application which uses ETL (cleaning processing data), traines a classifier and has a web application 
to get new data and print classification result, moreover shows plots (order from the most frequent category; histogram of categories per messages) from data of the database.


What this application is good for?

During a natural disaster there are a lot of messages sent from people to officiers via specific platforms or social media.
Since many different type of organizations work to help people, it is important to deliver the right message to the right organization.
Machine learning can help to classify messages using natural language processing and then it is easier to send it to the right place.



## File Descriptions <a name="files"></a>

In the data folder, there are two csv file containing messages and categories moreover the database file containing processed data.
Model in pickle file format is found in models file folder.
App folder contains the webapp files.


## Results<a name="results"></a>

How webapp works?

You can just type any message into the textbox and the output will be the highlighted categories which were most likely cover its content.

E.g. I need water here and there.  --> This message will have categories like water_supply, emergency.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight to make their disater data available. 
