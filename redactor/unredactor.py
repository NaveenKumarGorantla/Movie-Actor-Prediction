#python file
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
import re
import os
import glob


def get_traindata(input_files):
    if ( len(input_files) == 0):
        raise   Exception('empty values')
    list_filedata = []
    list_filepaths =[]
    #print(input_files)
    for eachfile in input_files:
        for filename in eachfile:
            filename = filename
            filepaths = glob.glob(str(filename))
            #print(filepaths)
            list_filepaths.append(filepaths)

    filepaths_list = nltk.flatten(list_filepaths)
    filepaths_list = filepaths_list[:200]
    print('Text files in Train data',len(filepaths_list))
    #print(filepaths_list)
    for filepath in filepaths_list:
        fileopen = open(filepath)
        file_data = fileopen.read()
        file_data = file_data.replace("<br />"," ")
        #file_data = re.sub(r'[^a-zA-Z0-9]'," ",file_data)
        list_filedata.append(file_data)
        fileopen.close()
    #print(list_filedata[0:10])
    return list_filedata

def get_testdata(input_files):
    if ( len(input_files) == 0):
        raise   Exception('empty values')
    list_filedata = []
    list_filepaths =[]
    for eachfile in input_files:
        for filename in eachfile:
            filename = filename
            filepaths = glob.glob(str(filename))
            list_filepaths.append(filepaths)

    filepaths_list = nltk.flatten(list_filepaths)
    filepaths_list = filepaths_list[1:2]
    print('Text files in Test data',len(filepaths_list))
    #print(filepaths_list)
    for filepath in filepaths_list:
        fileopen = open(filepath)
        file_data = fileopen.read()
        file_data = file_data.replace("<br />"," ")
        list_filedata.append(file_data)
        fileopen.close()
    return list_filedata


def get_features_unredacted_data(list_filedata):
    if ( len(list_filedata) == 0):
        raise Exception('Empty text data')
    Dictionary_list = []
    for data in list_filedata:

        words = nltk.word_tokenize(data)
        word_tokens = nltk.pos_tag(words)
        named_entities = nltk.ne_chunk(word_tokens)
    
        for entity in named_entities.subtrees():
            word_name =[]
            total_words = 0
            length_words = [];
            length_words = [ 0 for i in range(20)]
            i = 0
            if ( entity.label() == 'PERSON'):
                for leaf in entity.leaves():
                    word_name.append(leaf[0])
                    #print(leaf[0])
                    total_words += 1
                    length_words[i]= len( leaf[0])
                    i = i + 1
                word_name = ' '.join(word_name)
                #print(word_name)
                total_length = len(word_name)
                dict = {'F1': word_name,'F2': total_words, 'F3': length_words[0], 'F4': length_words[1], 'F5': length_words[2], 'F6': total_length, 'F7': max (length_words) ,
                        'F8': length_words[0]+ length_words[2], 'F9': length_words[1] + length_words[2],'F10': length_words[0] + length_words[1] }
                Dictionary_list.append(dict)
    #print(Dictionary_list[0:10])
    return Dictionary_list


def model_training():

    train_datapath = [['aclImdb/train/pos/*.txt']]
    train_data = get_traindata(train_datapath)

    if ( len(train_data) > 0):
        named_entities=get_features_unredacted_data(train_data)
        y_train = []
        x_train = []
        
        for i in named_entities:
            y_train.append(i['F1'])
            i.pop('F1')
            x_train.append(i)

        vectorizer = DictVectorizer(sparse=False)
        x_train = vectorizer.fit_transform(x_train)
        y_train = np.array(y_train)
        model= MultinomialNB()
        model.fit(x_train, y_train)

    else:
        return None

    return model


def get_redacted_data(list_filedata):
   
    if ( len(list_filedata) == 0):
        raise Exception('Empty text data')
    list_redacted_data = []
    for data in list_filedata:
        words = nltk.word_tokenize(data)
        word_tokens = nltk.pos_tag(words)
        named_entities = nltk.ne_chunk(word_tokens)
        list_person_names = []
        for entity in named_entities.subtrees():
            word_name =[]
            if ( entity.label() == 'PERSON'):
                for leaf in entity.leaves():
                    word_name.append(leaf[0])
                    
                word_name = ' '.join(word_name)
                list_person_names.append(word_name)
    
        set1 = list(set(list_person_names))
        print('set of name list :',set1)
        for person_name in set1:
            list_name=person_name.split()
            #print(list_name)
            pattern = '\u2588'
            if(len(list_name)==1):
                data = data.replace(person_name, pattern * len(list_name[0])+'$')
            elif(len(list_name) == 2):
                data = data.replace(person_name, pattern * len(list_name[0])+'$'+ pattern * len(list_name[1]))
            elif (len(list_name) == 3):
                data = data.replace(person_name, pattern * len(list_name[0]) +'$'+ pattern * len(list_name[1])+ '$' + pattern * len(list_name[2]))
            elif (len(list_name) == 4):
                data = data.replace(person_name, pattern * len(list_name[0]) + '$' + pattern * len(list_name[1])+ '$' + pattern * len(list_name[2])+ '$' + pattern * len(list_name[3]))
        
        list_redacted_data.append(data)
    return list_redacted_data




def get_features_redact_data(list_filedata):

    if ( len(list_filedata) == 0):
        raise Exception('Empty text data')
    redacted_names={}
    featureslist=[]
    pattern = '\u2588'
    for data in list_filedata:
        list_data = data.split(' ')
        for i in list_data:
            length_words=[]
            list_words=[]
            list_word=i.split('$')
            count=0
            length_words = [ 0 for i in range(5)]
            i = 0
            for word in list_word:
                if(word.startswith(pattern)):
                    list_words.append(word)
                    length_words[i] = len(word)
                    count+=1
                    i = i + 1
            full_name = ' '.join(list_words)
            length = len(full_name)
            #print('length of redacted words',length_words)
            if(count != 0):
                redacted_names = {'F1': full_name, 'F2': count, 'F3': length_words[0],'F4': length_words[1], 'F5':length_words[2],'F6': length, 'F7': max(length_words), 'F8':
                        length_words[0] + length_words[2], 'F9': length_words[1]+ length_words[2], 'F10': length_words[0] + length_words[1] }
                featureslist.append(redacted_names)
    return featureslist

def feature_prediction():

    model = model_training()
    testfiledata=get_testdata([['aclImdb/test/pos/*.txt']])
    print('unredacted test file data:',testfiledata)
    redacted_data=get_redacted_data(testfiledata)
    print('redacted test file data',redacted_data)
    features=get_features_redact_data(redacted_data)
    print('features of redacted data',features)

    vectorizer = DictVectorizer(sparse=False)
    x_test=[]
    y_test=[]

    for f in features:
        y_test.append(f['F1'])
        f.pop('F1')
        x_test.append(f)
    x_test = vectorizer.fit_transform(x_test)
    predicted_names =  model.predict(x_test)

    return predicted_names


def file_output(predicted_names):

    newfilepath = os.path.join(os.getcwd(),'outputfile')
    if not os.path.exists(newfilepath):
        os.makedirs(newfilepath)
        with open(os.path.join(newfilepath, 'output'), 'w') as outputfile:
            for item in y_pred:
                outputfile.write(item)
    elif os.path.exists(newfilepath):
        with open(os.path.join(newfilepath, 'output'), 'w') as outputfile:
            count = 1
            for name in predicted_names: 
                outputfile.write(str(count)+"." +" "+ name + " / ")
                count = count + 1



prediction = feature_prediction()
file_output(prediction)
