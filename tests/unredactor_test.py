#python file

import pytest
from redactor import unredactor


def test_get_trained_data():

    data = unredactor.get_traindata([['aclImd/pos/train/*.txt']])
    if(data is not None):
        assert True
    else:
        assert False


def test_get_test_data():
    data = unredactor.get_testdata([['aclImd/pos/test/*.txt']])
    if(data is not None):
        assert True
    else:
        assert False

def test_redact_data():
    text = [' Blue Sky' ,'Chris Morris', 'John']

    data = unredactor.get_redacted_data(text)
    if ( data):
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
