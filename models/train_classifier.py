import sys
import pandas as pd 
from sqlalchemy import create_engine
from nltk.tokenize import sent_tokenize

import re
import nltk
import pickle
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """Load database containing processed dataframe with message and category data

    Read the databse.
    Separates features and target variables; X (message column) and Y(categories: from 4th column until the end).
    Creates a list with category names from target columns' name.

    Args:
        database_filepath (str): path to the database.

    Returns:
        dataframes (feature and target arrays) and list (category names)

    """ 
    engine = create_engine('sqlite:///' + str(database_filepath))

    df = pd.read_sql_table('DisasterResponse', engine)
    
    X = df['message']
    
    Y = df.iloc[:,4:40]
    
    category_names = list(Y)
    
    return X,Y,category_names


def tokenize(text):
    """Process messages

    Separate each word.
    Makes words lower case, lemmatize (e.g. is->be) and remove stop words like -- the, of ...etc

    Args:
        text (str): path to the database.

    Returns:
        array containing processed text data

    """ 
    
    stop_words = stopwords.words("english")
    
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """Creates pipeline

    Pipeline steps:
        Convert a collection of text documents to a matrix of token counts.
        
        Transform a count matrix to a normalized tf or tf-idf representation.
        Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. 
        
        A random forest classifier.A random forest is a meta estimator that fits a number of decision tree classifiers
        on various sub-samples of the dataset and uses averaging to improve the
        predictive accuracy and control over-fitting.
  

    Args:
        None.

    Returns:
        Pipeline.
    """

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # comment out parameters which should be optimized
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],    
    }

    # if parameter optimization is aimed to be performed,choose new_model as return value
    new_model = GridSearchCV(model, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate training performance

    Prints mterics of prediction for each category
  

    Args:
        model: output of the build_model()
        X_test: test features
        Y_test: test targets
        category_names: names of the target column
        
    Returns: None.
    
        
    """
    
    # predict on test data
    Y_pred = model.predict(X_test)
    
    for i in range(0,len(category_names)):
 
    # printing the third element of the column
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    """Saving trained model to pickle file
  

    Args:
        model: output of the build_model()
        model_filepath (str): path of the saved model
        
    Returns: None.
    
    """"    
    pickle.dump(model, open(str(model_filepath), 'wb'))


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