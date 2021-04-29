#python file

import pytest
from redactor import unredactor
import sample2
from redactor import ac1Imdb

input_filepath = [['aclImdb/train/pos/*.txt']]

def test_get_traindata(input_filepath):

    train_data = get_traindata([['aclImdb/train/pos/*.txt']])
    if ( train_data):
        assert True
    else:
        assert False

