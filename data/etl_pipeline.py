#!usr/bin/env python 3

#libraries:

import pandas as pd 
from sqlalchemy import create_engine
import sys


def read_data(data):
    """
    this function takes a csv file 

    """
    df = pd.read_csv(data)
    return df

def merge_data(mes, cat):
    """
    merge datasets
    """
    df = mes.merge(cat, on="id")
    return df

def create_col(df):
    """
     Split categories into separate category columns
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
    df = df.drop(["categories"], axis=1)    
    df = pd.concat([df, cat], axis=1)
    #drop duplicates 
    df = df.drop_duplicates()
    return df

def create_database(df, db):
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
    print("db",db)
    print('db_name', db[:-3])
    create_database(df,db)
    print("database created successefully! ")










