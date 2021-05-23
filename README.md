# Disaster_Pipeline
 


## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
It is about classifying disaster messages. 


The project is divided into 3 sections : 

1/ ETL Pipeline : Given 2 datasets : messages and categories, we clean data and construct new dataset that we save it to an SQLite DB .

2/ ML Pipeline : Load dataset from database and build a machine learning pipeline to train and  classify text message in different categories.

3/ Deploy web app : Run the model in real time.

## Getting Started

### Settings

To clone the git repository:

```console
git clone https://github.com/MariumAZ/Disaster
```

To install Dependencies 

```console
cd Disaster_Pipeline 
pip3 install -r requirements.txt 
```

To run the app 

```console
cd app 
python3 run.py 
```

### Modifications :


To add modifications to the ETL pipeline and save changes : 
The etl_pipeline.py takes 3 arguments : 

- messages.csv
- categories.csv
- database where to save the processed dataframe 


```console
cd data 
python3 etl_pipeline.py messages.csv categories.csv <database_name>.db
```

To add modifications to the ML pipeline and save changes : 
The train_classifier.py takes 2 arguments :

- database name 
- name of the model to pickle 

```console
cd models
python3 train_classifier.py ./data/<database_name>.db  <model_name>.pkl
```




