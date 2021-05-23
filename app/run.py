#!usr/bin/env python 3
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens

#load data

engine = create_engine("sqlite:///../data/etl.db")
df = pd.read_sql_table("etl", engine)


#load model

model = joblib.load("../models/clf.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')


@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    
    dic = {}
    for col in df.columns[4:]:
        try:
            dic[col] = df[col].value_counts()[1]
        except:
            dic[col] = 0

    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}           
    dic_df = pd.DataFrame.from_dict(dic,orient='index')
    #print(dic_df.head())        
        




 # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=dic_df.index,     
                    y=dic_df[0]     
                )
            ],

            'layout': {
                'title': 'Distribution of Disasters',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "disaster categorie"
                }
            }
        }

        
    ]

    


    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()    






