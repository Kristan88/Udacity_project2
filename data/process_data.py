import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """load csv files containing messages and categories 

    Merges the two csv files via ids.

    Args:
        dataset (str): name of the csv data files

    Returns:
        dataframe

    """ 

    messages = pd.read_csv(str(messages_filepath))

    categories = pd.read_csv(str(categories_filepath))
    
    df = messages.merge(categories, on ='id')
    
    return df

def clean_data(df):
    """Add new category columns

    Creates new category columns from old category column's cell values.
    Sets the values ofeach column to 0 or 1.
    Deletes the old category column.

    Args:
        dataframe: output of the load_data()

    Returns:
        dataframe

    """ 
    
    categories = df['categories'].str.split(pat=';',expand=True)
    
    row = categories.iloc[0]
    
    #delete the last two characters of the string in each cell
    category_colnames = row.apply(lambda x: x[0:-2])
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1].replace({'2':'1'})
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
     
    df = df.drop(columns=['categories'])

    df = pd.concat([df, categories], axis=1).drop_duplicates()    
    
    return df


def save_data(df, database_filename):
    """Save processed data tp sql database

    Creates new sqlite engine with usergiven name.
    Load dataframe into newly created database.

    Args:
        dataframe: output of the clean_data()
        database_filename (str): name of the database

    Returns:
        dataframe

    """ 
    
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists = 'replace')    


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
