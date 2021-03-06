#python file

import pytest
from redactor import unredactor


def test_get_trained_data():

    data = unredactor.get_traindata([['aclImdb/train/pos/*.txt']])
    if(len(data)!= 0):
        assert True
    else:
        assert False


def test_get_test_data():
    data = unredactor.get_testdata([['aclImdb/test/pos/*.txt']])
    if(data is not None):
        assert True
    else:
        assert False

def test_redact_data():
    text = [' Blue Sky' ,'Chris Morris', 'John']

    data = unredactor.get_redacted_data(text)
    if ( len(data)!= 0):
        assert True
    else:
        assert False


def test_features_redact_data():
    text = ['████' ,'███████ ██ Chris ███ Morris', '███ John']
    data = unredactor.get_features_redact_data(text)
    if ( data):
        assert True
    else:
        assert False

def test_model():
    model = unredactor.model_training()
    if (model is not None):
        assert True
    else:
        assert False


