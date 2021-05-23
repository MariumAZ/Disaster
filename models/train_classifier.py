#!usr/bin/env python 3

#libraries:
import pandas as pd
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import sqlite3
import sys


def load_data(db_filepath):
    """
    load data from database
    """
    #engine = create_engine('sqlite:///{}'.format(db)

    engine = create_engine('sqlite:///' + db_filepath)
    db_name = db_path.split('/')[-1][:-3]
    df = pd.read_sql_table(db_name, engine)
    return df

def process_data(df):
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x)
    df = df.dropna()
    print(df.head())
    X  = df.iloc[:,1]
    y  = df.iloc[:,4:] 
    return X,y

def tokenize(text):
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens

def build_model():

    pipeline = Pipeline([
    
    ("count", CountVectorizer(tokenizer=tokenize)),
    ("tf", TfidfTransformer()),
    ("clf", MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {'clf__estimator__max_depth': [10, 20],
              'clf__estimator__max_leaf_nodes': [4, None]}
              #'clf__estimator__n_estimators': [10, 40]}


    model = GridSearchCV(pipeline, param_grid=parameters)     
    return pipeline

def evaluate_model(y_test,y_pred,target_names):
    print(classification_report(y_test.values, y_pred, target_names=target_names))

def save_model(model,model_path):
    """
    Save model to disk
    """
 
    with open(model_path, "wb") as f:
        pickle.dump(model,f)

if __name__ ==  "__main__":
    db_path, model_path = sys.argv[1:]   
    #extract db_name:
    print("loading data ...")
    df = load_data(db_path)
    X,y = process_data(df)
    target_names = list(y.columns.values)
    print("target names are ", target_names)
    x_train, x_test, y_train, y_test = train_test_split(X,y)
    model = build_model()
    #TODO add time for model fitting 
    print("fitting the model ...")
    model.fit(x_train,y_train)
    print("model evaluation ...")
    y_pred  = model.predict(x_test)
    evaluate_model(y_test,y_pred, target_names=target_names)
    print("saving to disk ...")
    save_model(model, model_path)

















    
