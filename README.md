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
7. 
Imports:
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize

Running the Program: Pipenv run python unredactor.py  ( command should be run under directory redactor)

Methods Implemented -

get_traindata :
This method takes the parameter which contains list of filepath of text files of train data. Inside the method, with the help of glob operator we are going to 
fetch each path of the text file that matches the wildcard in the glob operator. Out of many text files we are using only 200 text files to read and return
the data.

get_testdata :
This method takes the parameter which contains list of filepath of text files of train data. Inside the method, with the help of glob operator we are going to 
fetch each path of the text file that matches the wildcard in the glob operator. Out of many text files we are using only 1 text file to read and return
the data for testing purpose.

get_named_entities:
In this method we are extracting the named entities of the train data using named entity recognition. After extracting the names, we are adding the features of the 
named entities into a dictionary and return the dictionary containing the features.

model_training:
In this method we are train the model with the help of features of the training data which is returned from get_named_entities method. The dictionary which contains 
the features of training data is splitted into x_train and y_train( labels). we are using sklearn Naive bayes machine learning model for training and this
model is returned.

get_redacted_data: 
This method returns the redacted document data. In this method we use Named Entity recognition and identify the names of a Person.
All the person names identified are redacted with the help of unicode pattern and list of redacted document data is returned.

get_features_redact_data:
This method returns the dictionary which contains the features of the redacted document data. This method accepts the list of redacted data as a parameter
and we search for the hidden data in the each redacted document data. After finding the data hidden we define features of this data and add it to a
dictionary. These dictionary containing features of redacted data is returned from this method.

feature_prediction:
In this method we predict the data based on the features of the redacted data returned from get_features_redact_data. 
model_training method is invoked which returns the trained model, using this trained model we try to predict the names of the redacted test document 
data and return the list of predicted names.

file_output:
In this method we use the parameter which accepts the list of predicted names from feature_prediction method. We write all the predicted names into a new 
output text file by creating a new directory named output.




