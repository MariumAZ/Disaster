#!usr/bin/env python 3

import pandas as pd 
from sqlalchemy import create_engine
import sys


def read_data(data):
    """
    Reads a csv file 

    Parameters:
    data : csv_file

    Returns:
    df : pandas dataframe

    """
    df = pd.read_csv(data)
    return df

def merge_data(mes, cat):
    """
     Merges two datasets 

     Parameters:
     mes : dataframe
     cat : dataframe

     Returns :
     df : dataframe
    """
    df = mes.merge(cat, on="id")
    return df

def create_col(df):
    """
     Splits categories dataframe into separate categories columns
    """

    df = df["categories"].str.split(";",expand=True)    
    row = df.iloc[0].apply(lambda x:str(x)[:-2])
    category_colnames = list(row)
    df.columns = category_colnames
    for column in df:
        # set each value to be the last character of the string
        df[column] = df[column].apply(lambda x : str(x)[-1]).astype(str)
        # convert column from string to numeric
        df[column] = df[column].apply(lambda x:int(x))
    return df

def create_final(df, cat):
    """
    Processing final dataframe to save in the database
    - It concatenates two dataframes
    - Replaces the value 2 by 1 in the column related 

    Parameters :
    df, cat : dataframes

    Returns:
    df: final dataframe after processing 
    """


    df = df.drop(["categories"], axis=1)    
    df = pd.concat([df, cat], axis=1)
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x)
    #drop duplicates 
    df = df.drop_duplicates()
    #drop nan values
    df = df.dropna()

    return df

def create_database(df, db):
    """
    Creates a database
    Parameters: 
    df : dataframe
    db : database 
    """
    eng = create_engine('sqlite:///{}'.format(db))
    db_name = db[:-3]    
    df.to_sql(db_name, eng, index=False, if_exists='replace')

if __name__ == "__main__":

    messages, categories, db = sys.argv[1], sys.argv[2], sys.argv[3]
    messages = read_data(messages)
    categories = read_data(categories)
    df = merge_data(messages, categories)
    categories = create_col(categories)
    df = create_final(df, categories)
    print("Saving data to database ... ")
    create_database(df,db)
    print("database created successefully! ")










