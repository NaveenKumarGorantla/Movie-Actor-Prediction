# cs5293sp21-project2

The Unredactor

In this Project we are using the redacted documents which contains hidden sensitive information. Based on the information which is hidden we
are trying to predict the missing information with the help of a trained machine learning model.

Requuirements:
1. glob     
2. sklearn
3. nltk
4.  re 
5.  os
6. IMDB Movie Review dataset
Imports:
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize

Running the Program: Pipenv run python unredactor.py

Methods Implemented -

get_traindata : This method takes the parameter which contains list of filepath of text files of train data. Inside the method, with the help of glob operator we are going to 
fetch each path of the text file that matches the wildcard in the glob operator. Out of many text files we are using only 200 text files to read and return
the data.

get_testdata : This method takes the parameter which contains list of filepath of text files of train data. Inside the method, with the help of glob operator we are going to 
fetch each path of the text file that matches the wildcard in the glob operator. Out of many text files we are using only 1 text file to read and return
the data for testing purpose.

get_named_entities:
In this method we are extracting the named entities of the train data using named entity recognition. After extracting the names, we are adding the features of the 
named entities into a dictionary and return the dictionary containing the features.

model_training:
In this method we are train the model with the help of features of the training data which is returned from get_named_entities method. The dictionary which contains 
the features of training data is splitted into x_train and y_train( labels). we are using sklearn Naive bayes machine learning model for training and this
model is returned.







